import torch
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
            raise ValueError(f"Corrupted image file: {str(e)}")