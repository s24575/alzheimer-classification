import os
import shutil
from typing import Any

from fastapi import FastAPI, File, UploadFile
from predict import predict_image_from_path

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        predicted_class, confidence = predict_image_from_path(temp_path, "test")
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(temp_path)

    return {"class": predicted_class, "confidence": confidence}
