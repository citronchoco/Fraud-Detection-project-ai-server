import chromadb
from chromadb.utils import embedding_functions

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
fraud_collection = chroma_client.get_or_create_collection(name="fraud_cases", embedding_function=sentence_transformer_ef)

def find_similar_case(combined_text: str) -> str:
  rag_result = fraud_collection.query(
    query_texts=[combined_text],
    n_results=1
  )

  if rag_result['documents'] and rag_result['documents'][0]:
    return rag_result['documents'][0][0]
  return ""