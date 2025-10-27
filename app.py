import os
import torch
import shutil
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path
from src.download_static_esri_3tile_v2 import stitch_tiles
# import torch
import torchvision.models as models

# allow the ResNet class for safe unpickling
torch.serialization.add_safe_globals([models.resnet.ResNet])

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Substation Detection", layout="wide")
st.title("⚡ Substation Detection App")

st.markdown("Upload a CSV or Excel file with `Latitude` and `Longitude` columns:")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file is None:
    st.stop()

# -------------------------------
# Load input data
# -------------------------------
if uploaded_file.name.endswith(".csv"):
    df_org = pd.read_csv(uploaded_file)
else:
    df_org = pd.read_excel(uploaded_file)

if not {"Latitude", "Longitude"}.issubset(df_org.columns):
    st.error("Your file must contain columns: 'Latitude' and 'Longitude'")
    st.stop()

# Prepare dataframe
df = df_org.groupby(['Latitude', 'Longitude'], as_index=False).agg(lambda x: list(set(x)))
df["ResNet50_Result"] = "unknown"
df["ResNet18_Result"] = "unknown"
df["YOLO_Result"] = ""
df["YOLO_Confidence"] = 0.0
df["Prediction_Agree"] = False

# -------------------------------
# Model selection
# -------------------------------
st.sidebar.header("🧠 Model Configuration")
resnet50_path = st.sidebar.text_input("ResNet50 model path", "model/resnet50.pt")
resnet18_path = st.sidebar.text_input("ResNet18 model path", "model/resnet18.pt")
yolo_path     = st.sidebar.text_input("YOLOv8 model path", "model/yolo_best.pt")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models(resnet50_path, resnet18_path, yolo_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50_model = torch.load(resnet50_path, map_location=device)
    resnet18_model = torch.load(resnet18_path, map_location=device)
    resnet50_model.eval()
    resnet18_model.eval()
    yolo_model = YOLO(yolo_path)
    return resnet50_model, resnet18_model, yolo_model, device

resnet50_model, resnet18_model, yolo_model, device = load_models(resnet50_path, resnet18_path, yolo_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Temporary folders
IMAGE_DIR = Path("temp_images")
SORTED_DIR = Path("temp_sorted/disagreement")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SORTED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Run detection
# -------------------------------
if st.button("Run Detection"):
    progress = st.progress(0)
    for idx, row in df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        filename = f"{lat}_{lon}.png"
        img_path = IMAGE_DIR / filename

        # 1️⃣ Download image
        try:
            stitch_tiles(lat, lon, zoom=18, tile_count=3, tile_size=256, output_path=str(img_path))
        except Exception as e:
            st.warning(f"[{idx}] Failed to download image at ({lat}, {lon}): {e}")
            continue

        # 2️⃣ ResNet inference
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out50 = resnet50_model(img_tensor)
                out18 = resnet18_model(img_tensor)
            pred50 = torch.argmax(out50, dim=1).item()
            pred18 = torch.argmax(out18, dim=1).item()
            label50 = "substation" if pred50 == 1 else "not_substation"
            label18 = "substation" if pred18 == 1 else "not_substation"
            df.at[idx, "ResNet50_Result"] = label50
            df.at[idx, "ResNet18_Result"] = label18
        except Exception as e:
            st.warning(f"[{idx}] ResNet error: {e}")
            continue

        # 3️⃣ YOLOv8 detection
        try:
            results = yolo_model.predict(source=str(img_path), imgsz=768, conf=0.1, verbose=False)
            conf = float(results[0].boxes.conf.max()) if len(results[0].boxes) > 0 else 0.0
            label_yolo = "substation" if conf > 0.5 else "not_substation"
            df.at[idx, "YOLO_Result"] = label_yolo
            df.at[idx, "YOLO_Confidence"] = conf
        except Exception as e:
            st.warning(f"[{idx}] YOLO error: {e}")
            continue

        # 4️⃣ Agreement
        agree = (label50 == label18 == label_yolo)
        df.at[idx, "Prediction_Agree"] = agree
        if not agree:
            shutil.copy2(img_path, SORTED_DIR / filename)

        progress.progress((idx + 1) / len(df))

    st.success("✅ Detection complete!")

    # 5️⃣ Add map links
    def make_link(lat, lon):
        return f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
    df["Google Maps"] = df.apply(lambda r: make_link(r["Latitude"], r["Longitude"]), axis=1)

    # Show and export
    st.dataframe(df)
    output_path = "substation_predictions.csv"
    df.to_csv(output_path, index=False)
    with open(output_path, "rb") as f:
        st.download_button("📥 Download Results CSV", f, file_name="predictions.csv")
