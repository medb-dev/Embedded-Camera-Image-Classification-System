import requests
from PIL import Image
import io

def run_demo():
    url = "http://localhost:8000/api/v1/predict"
    image_path = "demo/sample_image.jpg"
    try:
        with open(image_path, "rb") as f: img_bytes = f.read()
        files = {'file': ('sample.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            res = response.json()
            print(f"Prediction: {res['class']} | Confidence: {res['confidence']*100:.2f}%")
            Image.open(io.BytesIO(img_bytes)).show()
        else: print(f"Error: {response.text}")
    except FileNotFoundError: print("Please add sample_image.jpg to demo/ folder")

if __name__ == "__main__":
    run_demo()