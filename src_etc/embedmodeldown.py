from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from pathlib import Path
import os

MODEL_PATH = r'C:\Dev\GitHub\ArgSearch\models\ko-sbert-nli'
#MODEL_NAME="jhgan/ko-sroberta-nli" 
#MODEL_CONFIG="c:/dev/github/Argsearch/models/ko-sbert-nli/config.json"
# "jhgan/ko-sroberta-multitask"
#"snunlp/KR-SBERT-V40K-klueNLI-augSTS"


#hf_hub_download(repo_id="jhgan/ko-sbert-nli", filename="config.json", cache_dir=MODEL_PATH, force_download=True)
model = SentenceTransformer('jhgan/ko-sroberta-nli')
model.save(MODEL_PATH)
config = AutoConfig.from_pretrained("jhgan/ko-sbert-nli")