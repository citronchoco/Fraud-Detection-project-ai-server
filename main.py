from typing import List
from fastapi import FastAPI, File, Form, UploadFile
import asyncio
from core.rag import find_similar_case
from core.vision import extract_text_from_buffer
from models.schemas import FraudResponse
import chromadb
from chromadb.utils import embedding_functions
import io
from PIL import Image
import numpy as np
import onnxruntime as ort

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
            
        # Vision 모듈 호출: 텍스트 추출
        extracted_texts = []
        for buffer in img_buffers:
            text = extract_text_from_buffer(buffer)
            extracted_texts.append(text)

        # RAG 모듈 호출: 추출된 텍스트 병합 후 유사 판례 찾기
        combined_text = " ".join(extracted_texts)
        reference_case = find_similar_case(combined_text)

        # LLM 돌리기

        #결과 반환
        return FraudResponse (
            status="FRAUD",
            fraudScore=85.5,
            description=f"유사 사례 참조: {reference_case[:20]}" if reference_case else "분석 완료"
        )
    