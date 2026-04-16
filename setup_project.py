import os

# Define the project structure and file contents
PROJECT_FILES = {
    "requirements.txt": """fastapi==0.109.0
uvicorn==0.27.0
torch==2.2.0
torchvision==0.17.0
pillow==10.2.0
numpy==1.26.0
matplotlib==3.8.0
httpx==0.26.0
python-multipart==0.0.9
pandas==2.2.0
seaborn==0.13.0""",

    "app/utils/decorators.py": """import time
import logging
import functools
from typing import Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("system_logger")

def log_execution(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"🚀 Executing {func.__name__}...")
        result = func(*args, **kwargs)
        logger.info(f"✅ Completed {func.__name__}")
        return result
    return wrapper

def timing(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"⏱️ {func.__name__} took {duration:.4f} seconds")
        return result, duration
    return wrapper""",

    "app/utils/validation.py": """from fastapi import HTTPException, UploadFile

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024 

def validate_image_file(file: UploadFile):
    extension = "." + file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")""",

    "app/services/image_processor.py": """import torch
from PIL import Image
import io
from torchvision import transforms
from app.utils.decorators import log_execution

class ImageProcessor:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @log_execution
    def process_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = self.transform(image)
            return tensor.unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Corrupted image file: {str(e)}")""",

    "app/services/inference_service.py": """import torch
import torch.nn.functional as F
from ml.model import SimpleCNN
from app.utils.decorators import log_execution, timing

class InferenceService:
    def __init__(self, model_path: str = "ml/model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        model = SimpleCNN().to(self.device)
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model
        except Exception:
            print(f"⚠️ Model file not found at {path}. Using untrained weights.")
            model.eval()
            return model

    @log_execution
    @timing
    def predict(self, image_tensor: torch.Tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, index = torch.max(probabilities, 1)
        return {"class": self.classes[index.item()], "confidence": round(confidence.item(), 4)}

    def get_prediction(self, image_tensor: torch.Tensor):
        result_tuple = self.predict(image_tensor)
        return result_tuple[0]""",

    "app/routes/predict.py": """from fastapi import APIRouter, UploadFile, File
from app.utils.validation import validate_image_file
from app.services.image_processor import ImageProcessor
from app.services.inference_service import InferenceService

router = APIRouter()
processor = ImageProcessor()
inference = InferenceService()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    validate_image_file(file)
    contents = await file.read()
    try:
        tensor = processor.process_image(contents)
        prediction = inference.get_prediction(tensor)
        return prediction
    except ValueError as ve:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(ve))""",

    "app/main.py": """from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI(title="Embedded Camera Image Classification API")
app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Image Classification API is running. Use /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)""",

    "ml/model.py": """import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x""",

    "ml/train.py": """import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(3):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "ml/model.pt")
    print("✅ Model saved to ml/model.pt")

if __name__ == "__main__":
    train()""",

    "ml/evaluate.py": """import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from model import SimpleCNN

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("ml/model.pt", map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_set.classes, yticklabels=test_set.classes)
    plt.savefig('ml/confusion_matrix.png')
    print("✅ Saved ml/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()""",

    "experiments/concurrency_test.py": """import asyncio
import httpx
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/api/v1/predict"
IMAGE_PATH = "demo/sample_image.jpg"
NUM_REQUESTS = 20

async def send_request(client, image_data):
    files = {'file': ('test.jpg', image_data, 'image/jpeg')}
    start = time.perf_counter()
    await client.post(API_URL, files=files)
    return time.perf_counter() - start

async def run_async_test(image_data):
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, image_data) for _ in range(NUM_REQUESTS)]
        start_time = time.perf_counter()
        await asyncio.gather(*tasks)
        return time.perf_counter() - start_time

def run_sequential_test(image_data):
    total_time = 0
    with httpx.Client() as client:
        for _ in range(NUM_REQUESTS):
            files = {'file': ('test.jpg', image_data, 'image/jpeg')}
            start = time.perf_counter()
            client.post(API_URL, files=files)
            total_time += (time.perf_counter() - start)
    return total_time

def run_threaded_test(image_data):
    def sync_request():
        with httpx.Client() as client:
            files = {'file': ('test.jpg', image_data, 'image/jpeg')}
            client.post(API_URL, files=files)
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(lambda _: sync_request(), range(NUM_REQUESTS)))
    return time.perf_counter() - start_time

async def main():
    try:
        with open(IMAGE_PATH, "rb") as f: image_data = f.read()
    except FileNotFoundError:
        print("❌ Sample image not found in demo/ folder."); return
    
    seq_time = run_sequential_test(image_data)
    thread_time = run_threaded_test(image_data)
    async_time = await run_async_test(image_data)

    df = pd.DataFrame({"Method": ["Sequential", "ThreadPool", "AsyncIO"], "Total Time (s)": [seq_time, thread_time, async_time]})
    print("\n--- PERFORMANCE RESULTS ---\n", df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())""",

    "demo/demo.py": """import requests
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
    run_demo()""",

    "Dockerfile": """FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]""",

    "README.md": """# Embedded Camera Image Classification System
- Train: `python ml/train.py`
- Run: `uvicorn app.main:app --reload`
- Test: `python experiments/concurrency_test.py`"""
}

def setup():
    print("🛠️ Starting Project Bootstrapping...")
    for path, content in PROJECT_FILES.items():
        # Create directories if they don't exist
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            print(f"✅ Created: {path}")
    
    print("\n🚀 Project created successfully!")
    print("\nNext steps:")
    print("1. pip install -r requirements.txt")
    print("2. python ml/train.py")
    print("3. uvicorn app.main:app --reload")
    print("4. (Optional) Place a .jpg image in demo/sample_image.jpg to test!")

if __name__ == "__main__":
    setup()
