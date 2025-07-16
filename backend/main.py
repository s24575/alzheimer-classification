from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.model_service import ModelService

MODEL_URI = "models:/alzheimer_classification/latest"


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


class ReloadModelRequest(BaseModel):
    model_uri: str

    model_config = {"json_schema_extra": {"example": {"model_uri": MODEL_URI}}}


class ReloadModelResponse(BaseModel):
    detail: str


app = FastAPI()
model_service = ModelService(MODEL_URI)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type not in ["image/png", "image/jpeg", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    try:
        image_bytes = await file.read()
        prediction, confidence = model_service.predict(image_bytes)
        return PredictionResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model(request: ReloadModelRequest) -> ReloadModelResponse:
    try:
        model_service.reload(request.model_uri)
        return ReloadModelResponse(
            detail=f"Model successfully reloaded from {request.model_uri}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")
