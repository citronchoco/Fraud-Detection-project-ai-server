import io
import easyocr
import numpy as np
from PIL import Image

# EasyOCR 리더 (한국어, 영어 지원 / 8GB 램을 위해 gpu=False 설정 가능)
# 모듈 로드 시 한 번만 메모리에 올립니다.
reader = easyocr.Reader(['ko', 'en'], gpu=False) 

def extract_text_from_buffer(buffer: io.BytesIO) -> str:
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    image_np = np.array(image)
    
    # OCR 텍스트 추출
    results = reader.readtext(image_np, detail=0)
    return " ".join(results)