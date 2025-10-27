import os
import math
import requests
from PIL import Image
import time

def latlon_to_tilexy(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile

def download_tile(x, y, zoom, save_path):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    return False

def stitch_tiles(lat, lon, zoom, tile_count=3, tile_size=256, output_path="stitched_image.png"):
    center_x, center_y = latlon_to_tilexy(lat, lon, zoom)
    half = tile_count // 2
    stitched = Image.new('RGB', (tile_count * tile_size, tile_count * tile_size))

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = center_x + dx
            y = center_y + dy
            tile_path = f"tile_{zoom}_{x}_{y}.png"
            success = download_tile(x, y, zoom, tile_path)
            if success:
                tile_img = Image.open(tile_path)
                stitched.paste(tile_img, ((dx + half) * tile_size, (dy + half) * tile_size))
                os.remove(tile_path)  # clean up
            else:
                print(f"Failed to download tile {x},{y}")
            time.sleep(0.2)  # avoid hammering the server

    stitched.save(output_path)
    print(f"Saved stitched image to {output_path}")

# === Example ===
##stitch_tiles(lat=25.937, lon=-97.5354, zoom=18, tile_count=3, output_path="dallas_zoom18_grid3x3.png")
##
