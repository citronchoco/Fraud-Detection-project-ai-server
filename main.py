from typing import List
import asyncio
import io

from fastapi import FastAPI, File, Form, UploadFile
from models.schemas import FraudResponse
from core.pipeline import analyze_chat_logic, analyze_image_manipulation

app = FastAPI()
concurrency_gate = asyncio.Semaphore(5)


@app.get("/test")
def test_api():
    return {"message": "파이썬 AI 서버가 성공적으로 켜졌습니다!"}


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(..., description="분석할 원본 이미지 파일 (JPEG/PNG)")
):
    image_data = await file.read()
    # AI 모델 분석 로직이 들어갈 자리
    return {
        "filename": file.filename,
        "description": "File received successfully",
        "status": "FRAUD",
        "fraudScore": 98.5,
    }


@app.post("/analyze", response_model=FraudResponse)
async def analyze_images(
    files: List[UploadFile] = File(...),
    scamType: str = Form(...),
    imageType: str = Form(...),
):
    async with concurrency_gate:
        # 이미지를 한 번만 읽고, BytesIO로 변환
        img_buffers: List[io.BytesIO] = []
        for file in files:
            img_bytes = await file.read()
            img_buffers.append(io.BytesIO(img_bytes))

        # 라우팅 1: 텍스트 기반 분석 (채팅 캡처)
        if imageType == "chat":
            return await analyze_chat_logic(img_buffers, scamType)

        # 라우팅 2: 시각 기반 분석 (프로필 사진 or 수익 인증샷)
        elif imageType in ["profile", "proof"]:
            return await analyze_image_manipulation(img_buffers, scamType, imageType)

        # 라우팅 3: 미지원 타입
        else:
            return FraudResponse(
                status="NORMAL",
                fraudScore=0.0,
                description="지원하지 않는 이미지 타입입니다.",
            )