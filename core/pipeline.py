import io
from typing import List

import google.generativeai as genai
from PIL import Image

from models.schemas import FraudResponse
from core.rag import find_similar_case
from core.vision import extract_text_from_buffer

genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini = genai.GenerativeModel("gemini-1.5-flash")


# ──────────────────────────────────────────────
# 라우트 1: 채팅 캡처 분석 (OCR → RAG → Gemini LLM)
# ──────────────────────────────────────────────
async def analyze_chat_logic(
    img_buffers: List[io.BytesIO],
    scam_type: str,
) -> FraudResponse:
    """
    채팅 캡처 이미지에서 텍스트를 추출(OCR)하고,
    RAG로 유사 판례를 검색한 뒤 Gemini로 최종 판단합니다.
    """
    # Step 1: OCR - 텍스트 추출
    # main.py에서 받은 버퍼를 그대로 사용
    extracted_texts = [extract_text_from_buffer(buf) for buf in img_buffers]
    combined_text = " ".join(extracted_texts)

    # Step 2: RAG - 유사 사기 판례 검색
    reference_case = find_similar_case(combined_text)

    # Step 3: 사기 유형별 분석 지침 생성
    if scam_type == "romance":
        analysis_guide = """
        로맨스 스캠 채팅 분석 지침:
        - 감정적 유대를 형성한 후 금전 요구로 전환되는 '피벗(Pivot)' 구간을 탐지하세요.
        - "급하게 돈이 필요하다", "비밀 투자처가 있다", "해외 송금" 등의 패턴을 찾으세요.
        - 참고 판례와의 유사도를 판단 근거에 포함하세요.
        """
    elif scam_type == "investment":
        analysis_guide = """
        투자 사기 채팅 분석 지침:
        - 원금 보장, 고수익 보장, 특정 계좌로의 입금 유도 등 전형적인 금융 사기 패턴을 찾으세요.
        - 리딩방·VIP방 초대, 특별 정보 제공 등의 미끼 멘트를 탐지하세요.
        - 참고 판례와의 유사도를 판단 근거에 포함하세요.
        """
    else:
        analysis_guide = "채팅 내용에서 사기 의심 패턴을 분석하세요."

    prompt = f"""당신은 디지털 금융 범죄 분석 전문가입니다.

[분석 대상 채팅 내용]
{combined_text}

[유사 사기 판례]
{reference_case if reference_case else "참고할 유사 판례가 없습니다."}

[분석 지침]
{analysis_guide}

위 내용을 종합하여 사기 여부를 판단하고, 결과를 JSON 형식으로 보고하세요."""

    response = await gemini.generate_content_async(
        contents=[prompt],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=FraudResponse,
        ),
    )
    return FraudResponse.model_validate_json(response.text)


# ──────────────────────────────────────────────
# 라우트 2: 이미지 시각 분석 (Gemini Vision)
# ──────────────────────────────────────────────
async def analyze_image_manipulation(
    img_buffers: List[io.BytesIO],
    scam_type: str,
    image_type: str,
) -> FraudResponse:
    """
    프로필 사진 또는 수익 인증샷을 Gemini Vision으로 분석합니다.
    OCR 없이 이미지 자체를 Gemini에 전달합니다.
    """
    # main.py에서 받은 BytesIO 버퍼를 PIL.Image로 변환
    pil_images: List[Image.Image] = []
    for buf in img_buffers:
        buf.seek(0)  # BytesIO 커서를 처음으로 되감기
        pil_images.append(Image.open(buf))

    # 사기 유형 × 이미지 유형 조합별 분석 지침
    if scam_type == "romance" and image_type == "profile":
        expert_role = "딥페이크 탐지 전문가"
        investigation_goal = """
        이 프로필 사진이 AI로 생성(Deepfake)되었거나 도용된 사진인지 분석하세요.
        - 눈동자 비대칭, 배경 왜곡, 피부 질감의 과도한 매끄러움 등 AI 생성물 특징을 찾으세요.
        - 비현실적인 화보형 사진인지 판단 근거에 포함하세요.
        """
    elif scam_type == "investment" and image_type == "proof":
        expert_role = "금융 범죄 수사관 및 이미지 조작 감식 전문가"
        investigation_goal = """
        이 수익 인증샷(그래프, 계좌 내역 등)의 조작 여부를 분석하세요.
        - 숫자 주변의 픽셀 번짐(포토샵 흔적), 폰트 불일치, 배경 그리드 정렬 오류를 찾으세요.
        - 표시된 수익률과 투자 금액이 수학적으로 논리적인지 검증하세요.
        """
    else:
        expert_role = "이미지 분석 전문가"
        investigation_goal = "이미지에 시각적 조작이나 모순점이 있는지 분석하세요."

    prompt = f"""당신은 {expert_role}입니다.

[수사 지침]
{investigation_goal}

위 지침에 따라 첨부된 이미지를 정밀 분석하고, 결과를 JSON 형식으로 보고하세요."""

    # ✅ Gemini Vision: 텍스트 프롬프트 + PIL 이미지 목록을 함께 전송
    response = await gemini.generate_content_async(
        contents=[prompt, *pil_images],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=FraudResponse,
        ),
    )
    return FraudResponse.model_validate_json(response.text)