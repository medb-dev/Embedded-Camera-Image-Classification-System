import torch
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
        return result_tuple[0]