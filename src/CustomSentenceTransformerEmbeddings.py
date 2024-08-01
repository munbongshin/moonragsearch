from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
import os
from pathlib import Path


#사용자 정의 임베딩 클래스를 만듭니다:
class CustomSentenceTransformerEmbeddings:
    def __init__(self, model_name_or_path=""):
        # 로컬 모델 경로 사용
        #project_root = Path(__file__).parent
        #env_path = project_root / MODEL_PATH
        #model_name_or_path = env_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(os.path.dirname(current_dir), "models", "ko-sbert-nli")
        if os.path.exists(MODEL_PATH):
            self.model = SentenceTransformer(MODEL_PATH)
        else:
            model = SentenceTransformer('jhgan/ko-sroberta-nli')
            model.save(MODEL_PATH)
            config = AutoConfig.from_pretrained("jhgan/ko-sbert-nli")
            if os.path.exists(MODEL_PATH):
                self.model = SentenceTransformer(MODEL_PATH)
                print(f"Model not found at {MODEL_PATH}. 임베딩 모델를 설치.")
            else:
                print(f"Model not found at {MODEL_PATH}. 임베딩 모델를 설치하세요.") 
        
        
    def embed_documents(self, documents):
        return self.model.encode(documents).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()