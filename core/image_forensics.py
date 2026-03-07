import io
import numpy as np
from PIL import Image, ImageChops
from models.schemas import FraudResponse

def detect_manipulation_ela(buffer: io.BytesIO) -> dict:
    """
    [investment scam] ELA(Error Level Analysis)를 이용해 이미지 합성/조작 흔적 찾기
    포토샵 등으로 숫자를 조작하면 해당 부분의 압축률이 달라지는 원리 이용
    """
    buffer.seek(0)
    original = Image.open(buffer).convert('RGB')
    
    # 임시로 낮은 품질로 저장했다가 다시 불러와서 차이(Error)를 계산
    temp_io = io.BytesIO()
    original.save(temp_io, 'JPEG', quality=90)
    temp_io.seek(0)
    compressed = Image.open(temp_io)
    
    diff = ImageChops.difference(original, compressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    diff = ImageChops.multiply(diff, scale)
    
    # 픽셀 차이의 평균값을 계산하여 조작 의심 점수 도출
    diff_np = np.array(diff)
    mean_error = float(np.mean(diff_np))
    
    is_manipulated = mean_error > 15.0  # 임계값 (테스트하며 조정 필요)
    
    return FraudResponse(
        status="SUSPICIOUS" if is_manipulated else "NORMAL",
        fraudScore=min(mean_error * 3, 99.0),
        description=f"이미지 압축률 오류 분석(ELA) 결과, 평균 오차값 {mean_error:.1f}로 조작 흔적이 {'발견되었습니다.' if is_manipulated else '보이지 않습니다.'}"
    )

def detect_deepfaㅔke_clip(buffer: io.BytesIO) -> dict:
    """
    [romance scam] Lab3 수업 자료의 CLIP을 응용하여 AI 생성 프로필인지 Zero-shot 분류
    """
    from transformers import CLIPProcessor, CLIPModel
    
    # 로컬에 다운로드된 가벼운 CLIP 모델 로드 (실무에서는 앱 시작 시 싱글톤으로)
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    
    labels = ["a real photo of a person", "an AI generated deepfake or highly manipulated portrait"]
    
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
    
    fake_prob = float(probs[1] * 100)
    
    return FraudResponse(
        status="FRAUD" if fake_prob > 70 else ("SUSPICIOUS" if fake_prob > 40 else "NORMAL"),
        fraudScore=fake_prob,
        description=f"CLIP 모델 시각 분석 결과, AI 생성물일 확률이 {fake_prob:.1f}% 입니다."
    )