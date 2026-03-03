from typing import List
from fastapi import FastAPI, File, Form, UploadFile
import asyncio
from models.schemas import FraudResponse
import chromadb
from chromadb.utils import embedding_functions
import io
from PIL import Image
import numpy as np


app = FastAPI()
concurrency_gate = asyncio.Semaphore(5)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
fraud_collection = chroma_client.get_or_create_collection(name="fraud_cases", embedding_function=sentence_transformer_ef)


@app.get("/test")
def test_api():
    return {"message": "야호! 유진님의 파이썬 AI 서버가 성공적으로 켜졌습니다!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(..., description="분석할 원본 이미지 파일 (JPEG/PNG)")):
    image_data = await file.read()

    # AI 모델 분석 로직이 들어갈 자리

    result_data = {
        "filename": file.filename,
        "description": "File received successfully",
        "status": "FRAUD",      # 사기 이미지 판별
        "fraudScore": 98.5     # 사기일 확률 98.5%
        }

    return result_data

@app.post("/analyze", response_model=FraudResponse)
async def analyze_images(
    files: List[UploadFile] = File(...),
    scamType: str = Form(...),
    imageType: str = Form(...)
):
    async with concurrency_gate:
        img_buffers = []
        for file in files:
            img_bytes = await file.read()
            img_buffers.append(io.BytesIO(img_bytes))
            
        extracted_texts = []
        for buffer in img_buffers:
            buffer.seek(0)
            original_img = Image.open(buffer).convert("RGB")

            resized_img = original_img.resize((224, 224))
            img_array = np.array(resized_img, dtype=np.float32)

            img_array /= 255.0
            

    # 결과 반환
    return {
        "status":"FRAUD",
        "fraudScore":85.5,
        "description":f"총 {len(files)}장의 {imageType} 데이터가 {scamType}유형으로 분석 완료되었습니다."
    }
    