import io
from typing import List
from ollama import AsyncClient

from models.schemas import FraudResponse
from core.rag import find_similar_case
from core.vision import extract_text_from_buffer
from core.image_forensics import detect_manipulation_ela, detect_deepfake_clip

OLLAMA_MODEL = "qwen2.5:1.5b"

async def analyze_chat_logic(
    img_buffers: List[io.BytesIO],
    scam_type: str,
) -> FraudResponse:
    # 1. 가벼운 로컬 OCR로 텍스트 추출
    extracted_texts = [extract_text_from_buffer(buf) for buf in img_buffers]
    combined_text = " ".join(extracted_texts)

    # 2. 공공 데이터 교차 검증 (RAG)
    reference_case = find_similar_case(combined_text)

    # 3. OCR 오타 감안 및 문맥 중심 파악 지침 추가
    prompt = f"""당신은 디지털 금융 범죄 분석 전문가입니다.
      다음 채팅 내용과 공공 사기 판례를 비교하여 사기 여부를 판단하세요.
      
      [채팅 내용]
      {combined_text}

      [공공기관 유사 사기 판례 (RAG 교차 검증 데이터)]
      {reference_case if reference_case else "참고할 유사 판례 없음."}

      [분석 유형]: {scam_type}

      [중요 지침]
      제공된 채팅 내용은 이미지에서 OCR 알고리즘으로 추출한 텍스트이므로 오타나 비문, 띄어쓰기 오류가 섞여 있을 수 있습니다. 개별 단어의 철자에 얽매이지 말고, 전체적인 문맥과 사기 수법의 흐름을 중심으로 판단하세요.

      반드시 아래 JSON 형식으로만 답변을 출력하세요. 마크다운이나 다른 텍스트는 절대 포함하지 마세요.
      {{
        "status": "FRAUD 또는 NORMAL 또는 SUSPICIOUS",
        "fraudScore": 0.0에서 100.0 사이의 숫자,
        "description": "분석 결과 요약 및 판단 근거 (유사 판례 일치 여부 포함)"
      }}
      """
    # 4. AsyncClient를 사용한 비동기 논블로킹 호출
    client = AsyncClient()
    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json", 
    )
    
    # 5. Pydantic 모델로 파싱 및 검증
    return FraudResponse.model_validate_json(response['message']['content'])


async def analyze_image_manipulation(
    img_buffers: List[io.BytesIO],
    scam_type: str,
    image_type: str,
) -> FraudResponse:
    
    highest_fraud_response = None
    max_score = -1.0

    # 💡 피드백 반영: 여러 장의 이미지를 순회하며 가장 의심스러운(점수가 높은) 결과를 채택
    for buf in img_buffers:
        if scam_type == "investment" and image_type == "proof":
            current_response = detect_manipulation_ela(buf)
        elif scam_type == "romance" and image_type == "profile":
            current_response = detect_deepfake_clip(buf)
        else:
            return FraudResponse(
                status="NORMAL", 
                fraudScore=0.0, 
                description="분석을 지원하지 않는 이미지 타입입니다."
            )
        
        # 최댓값 갱신 (가장 사기 확률이 높은 결과 저장)
        if current_response.fraudScore > max_score:
            max_score = current_response.fraudScore
            highest_fraud_response = current_response

    # 만약 에러 등으로 버퍼가 비어있었다면 기본값 반환
    if not highest_fraud_response:
        return FraudResponse(status="NORMAL", fraudScore=0.0, description="분석할 이미지가 없습니다.")

    return highest_fraud_response