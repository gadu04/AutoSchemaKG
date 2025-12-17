import gdown
import os

url = "https://drive.google.com/drive/folders/1BVAxWW0mxveNBLM1ZIU0aFcCSxT7ijSt?hl=vi"

output_folder = "data"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Downloading files from Drive to '{output_folder}' directory...")

gdown.download_folder(url, output=output_folder, quiet=False, use_cookies=False)

print("\nAll data downloaded successfully!")