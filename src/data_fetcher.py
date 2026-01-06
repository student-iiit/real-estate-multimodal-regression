import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

if MAPBOX_TOKEN is None:
    raise ValueError("MAPBOX_TOKEN not found. Check your .env file.")

# Image parameters
IMAGE_SIZE = 224        # 224x224 pixels (standard for CNNs)
ZOOM_LEVEL = 18         # Controls how close the satellite view is
STYLE = "mapbox/satellite-v9"

# Rate limiting (to avoid API bans)
SLEEP_TIME = 0.2        # seconds between requests

def fetch_satellite_image(lat, lon, save_path):
    """
    Download a satellite image for given latitude and longitude.
    Returns True if successful, False otherwise.
    """

    url = (
        f"https://api.mapbox.com/styles/v1/{STYLE}/static/"
        f"{lon},{lat},{ZOOM_LEVEL}/"
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            return False

    except requests.exceptions.RequestException:
        return False

def download_images_from_dataframe(df, split_name):
    """
    Download satellite images for all rows in a dataframe.
    """

    output_dir = f"data/images/{split_name}"
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        house_id = row["id"]
        lat = row["lat"]
        lon = row["long"]

        save_path = os.path.join(output_dir, f"{house_id}.jpg")

        # Skip if image already exists
        if os.path.exists(save_path):
            continue

        success = fetch_satellite_image(lat, lon, save_path)

        if success:
            success_count += 1

        time.sleep(SLEEP_TIME)

    print(f"Downloaded {success_count} images to {output_dir}")


if __name__ == "__main__":

    # Load datasets
    train_df = pd.read_excel("data/raw/train.xlsx")
    test_df = pd.read_excel("data/raw/test.xlsx")

    print("Downloading TRAIN images...")
    download_images_from_dataframe(train_df, "train")

    print("Downloading TEST images...")
    download_images_from_dataframe(test_df, "test")

    print("Image download completed.")
