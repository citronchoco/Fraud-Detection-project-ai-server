import chromadb
from chromadb.utils import embedding_functions

# 모듈 레벨 싱글톤: 앱 시작 시 1회만 초기화
_sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_chroma_client = chromadb.Client()
_fraud_collection = _chroma_client.get_or_create_collection(
    name="fraud_cases",
    embedding_function=_sentence_transformer_ef,
)


def find_similar_case(combined_text: str) -> str:
    """추출된 채팅 텍스트와 가장 유사한 사기 판례를 검색합니다."""
    rag_result = _fraud_collection.query(
        query_texts=[combined_text],
        n_results=1,
    )

    if rag_result["documents"] and rag_result["documents"][0]:
        return rag_result["documents"][0][0]
    return ""