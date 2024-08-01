from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from pathlib import Path

MODEL_PATH = "../models/ko-sbert-nli"
MODEL_NAME="jhgan/ko-sroberta-nli" 
MODEL_CONFIG = "../models/ko-sbert-nli/config.json"
# "jhgan/ko-sroberta-multitask"
#"snunlp/KR-SBERT-V40K-klueNLI-augSTS"
MODEL_CONFIG="./models/ko-sbert-nli/config.json"

hf_hub_download(repo_id="jhgan/ko-sbert-nli", filename="config.json", cache_dir="./models/ko-sbert-nli")
config = AutoConfig.from_pretrained(MODEL_CONFIG)