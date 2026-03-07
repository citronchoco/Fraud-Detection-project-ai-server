import sys
import os

# 상위 폴더의 core 모듈을 불러오기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import fraud_collection

def ingest_real_huggingface_data():
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets 라이브러리가 없습니다. 터미널에서 'pip install datasets'를 실행해주세요.")
        return

    print("Hugging Face 오픈소스 한국어 사기/스팸 데이터셋을 다운로드합니다...")
    dataset = load_dataset("meal-bbang/Korean_message", split="train")
    
    documents = []
    metadatas = []
    ids = []
    
    print("실제 사기/스팸 데이터만 추출하는 중...")
    
    fraud_cases = [row for row in dataset if row['class'] == 2]
    
    # 최대 1000건만 추출 (ChromaDB 과부하 방지)
    limit = min(1000, len(fraud_cases))
    
    for i in range(limit):
        content_text = fraud_cases[i]['content']
        
        if content_text and str(content_text).strip():
            documents.append(str(content_text))
            metadatas.append({
                "source": "HuggingFace_Korean_message",
                "type": "real_spam_smishing"
            })
            ids.append(f"real_hf_case_{i}")

    if not documents:
        print("데이터를 추출하지 못했습니다.")
        return

    print(f"총 {len(documents)}건의 100% 리얼 사기 데이터가 준비되었습니다!")
    print("ChromaDB 창고에 영구 적재합니다...")
    
    fraud_collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("리얼 공개 데이터 RAG 적재가 완벽하게 완료되었습니다!")

if __name__ == "__main__":
    ingest_real_huggingface_data()