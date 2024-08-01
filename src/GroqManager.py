import os
import streamlit as st
from typing import List, Dict, Any, Union
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
from langchain.schema import Document

class ChatMessage(BaseModel):
    role: str
    content: str

class GroqManager:
    def __init__(self):
        self.model = "Llama-3.1-70b-Versatile"
        
        # 프로젝트 루트 디렉토리 경로 설정
        project_root = Path(__file__).parent.parent

        # .env 파일의 경로 설정
        env_path = project_root / '.env'

        # .env 파일 로드
        load_dotenv(dotenv_path=env_path)
                
        # GROQ_API_KEY 환경 변수에서 API 키 가져오기
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # Groq 클라이언트 초기화
        self.client = Groq(api_key=self.api_key)

    def set_model(self, modelname):
        self.model = modelname
    
    def generate_response(self, docs: str, query: str):
        try:
            prompt = f"""Context:
{docs}

User Question: {query}

Please provide a detailed answer based on the given context:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant. "
                     "You always provide well-reasoned answers that are both correct and helpful."
                     "Use the following pieces of context to answer the user's question."
                     "If you don't know the answer, just say that you don't know, "
                     "don't try to make up an answer. Please answer in Korean."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2048
            )

            answer = response.choices[0].message.content
            
            return {
                "content": answer,
                "metadata": {
                    "model": self.model,
                    "usage": response.usage.dict()
                }
            }
        except Exception as e:
            return {"content": f"Error: {str(e)}", "metadata": {"error": str(e), "model": self.model}}

@st.cache_data
def get_groq_response(_groq_manager, _docs: Union[List[Document], List[str], str], query: str):
    if isinstance(_docs, list):
        if all(isinstance(doc, Document) for doc in _docs):
            docs_str = "\n".join(doc.page_content for doc in _docs)
        elif all(isinstance(doc, str) for doc in _docs):
            docs_str = "\n".join(_docs)
        else:
            raise ValueError("docs must be a list of Document objects or strings")
    elif isinstance(_docs, str):
        docs_str = _docs
    else:
        raise ValueError("docs must be a list of Document objects, a list of strings, or a string")

    response = _groq_manager.generate_response(docs_str, query)
        
    # content가 문자열인지 확인하고, 아니라면 문자열로 변환
    if not isinstance(response['content'], str):
        response['content'] = str(response['content'])

    return response

def main():
    st.title("Groq를 이용한 Q&A 시스템")

    groq_manager = GroqManager()

    docs = st.text_area("문서 내용을 입력하세요:", height=200)
    query = st.text_input("질문을 입력하세요:")

    if st.button("답변 생성"):
        if docs and query:
            with st.spinner('AI 답변을 생성 중...'):
                docs_list = [docs] if isinstance(docs, str) else docs
                result = get_groq_response(groq_manager, docs_list, query)
            
            st.write(f"AI 답변: {result['content']}")
            st.write(f"사용된 모델: {result['metadata'].get('model', 'Unknown')}")
            if 'usage' in result['metadata']:
                st.write(f"토큰 사용량: {result['metadata']['usage']}")
            if 'error' in result['metadata']:
                st.error(f"오류 발생: {result['metadata']['error']}")
        else:
            st.warning("문서 내용과 질문을 모두 입력해주세요.")

if __name__ == "__main__":
    main()