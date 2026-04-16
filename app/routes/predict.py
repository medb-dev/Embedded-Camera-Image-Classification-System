from fastapi import APIRouter, UploadFile, File
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
        raise HTTPException(status_code=400, detail=str(ve))