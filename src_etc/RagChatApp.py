from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import ChatMessage
from langchain_core.documents import Document
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain_core.embeddings import Embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

import logging, os, platform, base64, time
import  subprocess, requests
from tkinter import Tk, filedialog
# managechromadb0k2.py에서 ChromaDBManager를 import
from ChromaDbApp import ChromaDBManager
from typing import IO 

MODEL_PATH="C:/Dev/models/ko-sbert-nli"
ARG_TEMP_PATH="c:/tmp"

class CustomOpenAIEmbeddings(Embeddings):
    def __init__(self, client, model_name_or_path=MODEL_PATH):
        self.client = client
        self.model = SentenceTransformer(model_name_or_path)

    def embed_documents(self, texts):
        embeddings = self.client.embeddings.create(model=self.model, input=texts)
        return [embedding.embedding for embedding in embeddings.data]

    def embed_query(self, text):
        embedding = self.client.embeddings.create(model=self.model, input=[text])
        return embedding.data[0].embedding
    
#사용자 정의 임베딩 클래스를 만듭니다:
class CustomSentenceTransformerEmbeddings:
    def __init__(self, model_name_or_path=MODEL_PATH):
        # 로컬 모델 경로 사용
        if os.path.exists(model_name_or_path):
            self.model = SentenceTransformer(model_name_or_path)
        else:
            raise ValueError(f"Model not found at {model_name_or_path}. Please download the model first.")
    
    def embed_documents(self, documents):
        return self.model.encode(documents).tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()

# OpenAI 클라이언트 초기화
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 커스텀 임베딩 객체 생성
#embeddings = CustomOpenAIEmbeddings(client, "heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF")
#embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
embeddings = CustomSentenceTransformerEmbeddings(MODEL_PATH)

# LLM을 전역 변수로 초기화
lm_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="QuantFactory/EEVE-Korean-Instruct-10.8B-v1.0-GGUF",
    streaming=True,
    temperature=0,
)

ola_llm =  Ollama(model="llama3:instruct")

collection_name = "rag_test"
FILLTERED_DOC_NUMBER = 5
BASE_URL = "http://localhost:1234/v1"
DEFAULT_LLMNAME = "QuantFactory/EEVE-Korean-Instruct-10.8B-v1.0-GGUF"
# Prompt template
raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] Use the following pieces of context to answer the user's question into korean.\n
     If you don't know the answer, just say that you don't know, don't try to make up an answer.\n
     add the summarization of answer at the end of the answer.[/INST]</s>
    [INST] Context: {context}
    Question: {question}
    Answer:
    [/INST]
    """
)


class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

class RAGChatApp:
    def __init__(self):
        load_dotenv()
        self.persist_directory = "c:/DEV/mooland/PDFchatLLM/chroma_db"
        self.collection_name = "rag_test"
        self.db_manager = ChromaDBManager(persist_directory=self.persist_directory)
        self.llm = lm_llm
        self.initialize_session_state()
        os.environ["POSTHOG_DISABLED"] = "1"
        
        
    def get_ollama_models(self):
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]  # 첫 줄(헤더)을 제외
        models = [line.split()[0] for line in lines]  # 모델 이름만 추출
        return models
    
    def get_lm_studio_models(self, base_url):
        try:
            response = requests.get(f"{base_url}/model", timeout=5)  # 5초 타임아웃 추가
            if response.status_code == 200:
                model_info = response.json()
                # 'id' 필드가 있다면 그것을 사용하고, 없다면 전체 응답을 문자열로 반환
                model_name = model_info.get('id', str(model_info))
                return model_name  # 문자열로 반환
            else:
                st.error(f"LM Studio 서버 응답 오류: {response.status_code}")
                return DEFAULT_LLMNAME
        except requests.RequestException as e:
            if "NewConnectionError" in str(e):
                st.error(f"LM Studio 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
            elif "ConnectTimeoutError" in str(e):
                st.error(f"LM Studio 서버 연결 시간 초과. 서버 주소와 포트를 확인해주세요.")
            else:
                st.error(f"LM Studio 서버 연결 오류가 발생했습니다.")
            st.error(f"상세 오류: {e}")
            return DEFAULT_LLMNAME
           
    def set_llm_model(self, modelname):
        self.llm = modelname

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "docs" not in st.session_state:
            st.session_state.docs = None
        if "selected_sources" not in st.session_state:
            st.session_state.selected_sources = []
        if "last_query" not in st.session_state:
            st.session_state.last_query = ""
        if "rag_usage" not in st.session_state:
            st.session_state.rag_usage = "Rag+LLM"
        if "max_docnum" not in st.session_state:
            st.session_state.max_docnum = 3
        if "apply_filter" not in st.session_state:
            st.session_state.apply_filter = False
        if "filtered_sources" not in st.session_state:
            st.session_state.filtered_sources = []
        if "Rag+LLM" not in st.session_state:    
            st.session_state.RagUsage = "Rag+LLM"
            
    def set_collection_name(self, ragname):
        self.collection_name = ragname
            
    def parse_input(self, input_text):
        # 입력 문자열을 분리
        parts = input_text.split(']', 1)
        
        if len(parts) > 1 and parts[0].startswith('['):
            # 예약어가 있는 경우
            keyword = parts[0][1:].strip().lower()  # '[' 제거 및 소문자 변환
            query = parts[1].strip()
            return keyword, query
        else:
            # 예약어가 없는 경우
            return None, input_text.strip()
    
    def filter_results_by_score(results, threshold=400):
        """
        results 리스트에서 score가 threshold 이하인 항목만 필터링하여 반환합니다.
        
        :param results: 검색 결과 리스트. 각 항목은 'score' 키를 포함하는 딕셔너리여야 합니다.
        :param threshold: 필터링할 score의 임계값. 기본값은 400입니다.
        :return: 필터링된 결과 리스트
            """
        filtered_results = [item for item in results if item.get('score', float('inf')) <= threshold]
        return filtered_results
    
    def perform_search(self,query, db_manager, collection_name, selected_sources=None):
        raw_results = db_manager.search_collection(collection_name, query, n_results=FILLTERED_DOC_NUMBER,inscore=350)
        
        if not raw_results:
            return []  # 검색 결과가 없으면 빈 리스트 반환
        
        # 결과를 Document 객체로 변환하고 관련도 점수에 따라 정렬
        docs = [
            Document(
                page_content=result['page_content'],
                metadata={"score": result['score'], **result['metadata']}
            )
            for result in raw_results
        ]
        
        
        #docs.sort(key=lambda x: x.metadata['score'], reverse=False)
        
        #threshold = 500
        #filtered_docs = [doc for doc in docs if doc.metadata['score'] <= threshold]
        filtered_docs = [doc for doc in docs]
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting Perform Search")
        logger.info(f"filtered Doc: {filtered_docs}")
        if selected_sources:
            filtered_docs = [doc for doc in filtered_docs if doc.metadata.get('source', 'Unknown') in selected_sources]
        
        return filtered_docs


    def setup_ui(self):
        st.set_page_config(page_title="AI에게 질문하기", layout="wide")
        st.header("Ask your PDF, DOC, PPT, Excel, HWP 💬")
        
        # 메인 컨텐츠 영역 생성
        main_content = st.empty()
        
        # 메인 컨텐츠 안에 두 개의 컬럼 생성
        # 두 개의 컬럼 생성
        global maincol1, maincol2
        with main_content.container():
            maincol1, maincol2 = st.columns([7, 3])  # 7:3 비율로 분할

            
    def setup_sidebar(self):
        with st.sidebar:
            st.header("Rag Menu")
            
            # 탭 생성
            tab1, tab2, tab3, tab4 = st.tabs(["Arg Chat","임베딩", "ChromaDB 관리","LLM 모델"])
            
            with tab1:                
                self.setup_arg_chat_tab()
            
            with tab2:
                self.setup_embed_page()
            
            with tab3:                
                self.setup_chromadb_tab()
                
            with tab4:                
                self.setup_llmmodel_page()
        
    def setup_arg_chat_tab(self):
        st.header("Arg Chat")
        #st.session_state.max_docnum = st.number_input("참조문서 수 입력", max_value=20, min_value=3, step=1, value=st.session_state.max_docnum)
        st.session_state.rag_usage = st.radio("Rag이용여부 👇", ["Rag+LLM", "LLM"], index=0 if st.session_state.rag_usage == "Rag+LLM" else 1)
        collections = self.db_manager.list_collections()
        selected_collection = st.selectbox("검색할 컬렉션을 선택하세요", collections, key="search_collection_name")
        self.set_collection_name(selected_collection)
        self.setup_document_selection(selected_collection)
        
    def display_search_results(self, results):
        with maincol2:
            if not results:
                st.write("No results found.")
            else:
                for i, result in enumerate(results, 1):
                    key = "view_" + str(i)
                    with st.expander(f"Result {i}"):
                        if isinstance(result, dict):
                            st.write(f"**Content:** {result['page_content']}")
                            st.write(f"**Metadata:** {result['metadata']}")
                            st.write(f"**Score:** {result['score']:.1f}")
                            if st.button(f"**File_name:** {result['metadata']['file_name']}", key=key):
                                # 파일을 불러오는 기능 구현                        
                                #self.open_file_in_streamlit(result['original_file']['filename'])
                                self.show_pdf(result['metadata']['file_name'])
                        else:
                            st.write(f"**Content:** {result.page_content}")
                            st.write(f"**Metadata:** {result.metadata}")
                            st.write(f"**Score:** {result.metadata.get('score'):.1f}")
                            # 파일 이름을 클릭하면 해당 문서를 불러오기
                            if st.button(f"**File_name:** {result.metadata.get('file_name')}", key=key):
                                # 파일을 불러오는 기능 구현                        
                                #self.open_file_in_streamlit(result['original_file']['filename'])
                                self.show_pdf(result.metadata.get('file_name'))
                        
    def show_pdf(self, file_path):
        logging.basicConfig(level=logging.INFO)
        st.write("로깅 테스트")
        logging.info("이 메시지가 콘솔에 표시되어야 합니다.")
        st.set_option('client.showErrorDetails', True)
        placeholder = st.empty()
        placeholder.text("이 텍스트가 보이나요?")
        file_path = os.path.join(self.persist_directory,file_path)
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    def open_file_in_streamlit(self, file_path):
        file_path = os.path.join(self.persist_directory,file_path)
        if os.path.exists(file_path):
            file_extension = os.path.splitext(file_path)[1].lower()
            supported_extensions = ['.pdf', '.doc', '.docx', '.hwp','.pptx','.ppt','.xls','.xlsx']
            
            if file_extension in supported_extensions:
                try:
                    if platform.system() == 'Windows':
                        os.startfile(file_path)
                        st.success(f"{file_path} 파일이 기본 애플리케이션에서 열렸습니다.")
                    else:
                        st.error("이 기능은 Windows에서만 지원됩니다.")
                except Exception as e:
                    st.error(f"파일을 열지 못했습니다: {str(e)}")
            else:
                st.warning(f"지원되지 않는 파일 형식입니다: {file_extension}")
                st.write("지원되는 파일 형식: .pdf, .doc, .docx, .hwp")
        else:
            st.error(f"파일을 찾을 수 없습니다: {file_path}")
                        
    def list_files_in_directory(self, directory):
        files = []
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                files.append(file)
        return files

    def setup_document_selection(self,collection_name):
        st.subheader("검색대상 문서선택")
        
        # 세션 상태 초기화
        if 'filtered_sources' not in st.session_state:
            st.session_state.filtered_sources = []
        if 'selected_sources' not in st.session_state:
            st.session_state.selected_sources = []
        if 'show_search_results' not in st.session_state:
            st.session_state.show_search_results = False
        if 'apply_filter' not in st.session_state:
            st.session_state.apply_filter = False

        # 검색 입력 필드
        source_search = st.text_input("문서검색:", key="source_search_input")
        
        # 검색 버튼 검색어가 없으면 모든 파일명을 반환함
        if st.button("검색"):
            st.session_state.filtered_sources = self.db_manager.get_all_documents_source(collection_name, source_search)
            st.session_state.show_search_results = True
        
        # 검색 결과를 대화상자 형태로 표시
        if st.session_state.show_search_results:
            with st.expander("검색 결과", expanded=True):
                st.write("다음 문서들이 검색되었습니다. 선택하세요:")
                for source in st.session_state.filtered_sources:
                    if st.checkbox(source, key=f"check_{source}"):
                        if source not in st.session_state.selected_sources:
                            st.session_state.selected_sources.append(source)
                    elif source in st.session_state.selected_sources:
                        st.session_state.selected_sources.remove(source)                
                if st.button("선택 완료"):
                    st.session_state.show_search_results = False
                    
        
        # 선택된 문서 표시
        selected = st.multiselect(
            "선택된 문서:", 
            options=st.session_state.filtered_sources,
            default=st.session_state.selected_sources,
            key="final_selected_sources",
        )
        
        # multiselect의 결과로 selected_sources 업데이트
        st.session_state.selected_sources = selected
        
        col1, col2 = st.columns(2)
        with col1: 
            st.session_state.apply_filter = st.checkbox("적용", value=st.session_state.apply_filter)
        with col2:
            if st.button("필터 초기화"):
                st.session_state.selected_sources = []
                st.session_state.filtered_sources = []
                st.session_state.apply_filter = False
                
        pass
            
    def setup_chromadb_tab(self):
        st.title("Arg Search를 위한 ChromaDB 관리 Section")
        # get_collection_name 메서드 호출
        
        management_menu = st.selectbox(
        "작업을 선택하세요",
        ("Collection 생성", "Collection 삭제", "Collection 내용검색","Collection 내용보기"), key="tabmenu_select"
        )
        collections = self.db_manager.get_list_collections()
        if not isinstance(collections, (list, tuple)):
            st.error("컬렉션 목록을 가져오는 데 문제가 발생했습니다.")
            return
        if management_menu == "Collection 생성":
            st.header("새 컬렉션 생성")
            collection_name = st.text_input("생성할 컬렉션 이름을 입력하세요")                
            if st.button("생성", key="create_new_collection"):
                # 여기에 ChromaDB 컬렉션 생성 로직 추가
                # collection_name이 collections에 있는지 확인
                if collection_name not in collections:
                    result = self.db_manager.create_collection(collection_name)
                    st.info(f"컬렉션 '{result}'이 생성되었습니다.")
                else:
                    st.info(f"컬렉션 '{collection_name}'이 존재하여 생성할 수 없습니다.")
        elif management_menu == "Collection 삭제":
            st.header("컬렉션 삭제")
            # 상태 초기화
            if 'delete_state' not in st.session_state:
                st.session_state.delete_state = 'initial'

            selected_collection = st.selectbox("삭제할 컬렉션을 선택하세요", collections, key="delete_collection_select")

            if st.session_state.delete_state == 'initial':
                if st.button("삭제", key="delete_collection"):
                    st.session_state.delete_state = 'confirm'
                    
            elif st.session_state.delete_state == 'confirm':
                st.write(f"'{selected_collection}' 컬렉션을 삭제하시겠습니까?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("예", key="confirm_yes"):
                        # 삭제 로직 실행
                        result = self.db_manager.delete_collection(selected_collection)
                        st.success(f"'{selected_collection}' 컬렉션이 삭제되었습니다.")
                        st.session_state.delete_state = 'initial'
                        
                with col2:
                    if st.button("아니오", key="confirm_no"):
                        st.info("삭제가 취소되었습니다.")
                        st.session_state.delete_state = 'initial'
                        
        elif management_menu == "Collection 내용검색":
            st.header("컬렉션 검색")
            #collections = self.db_manager.list_collections()
            # 컬렉션이 리스트나 튜플 형태인지 확인
            selected_collection = st.selectbox("검색할 컬렉션을 선택하세요", collections, key="search_collection_select")
        
            search_query = st.text_input("검색어를 입력하세요")
            if st.button("검색", key="search_collection_content"):
                # 여기에 검색 로직 추가
                results = self.db_manager.search_collection(selected_collection, search_query,self.db_manager.docnum,inscore=350)                
                self.display_search_results(results)                
        elif management_menu == "Collection 내용보기":
            st.header("컬렉션 내용보기")
            collections = self.db_manager.list_collections()       
            selected_collection = st.selectbox("컬렉션을 선택하세요", collections,  key="view_collection_select")            
            if st.button("내용 보기", key="view_collection_content"):
                # 여기에 선택된 컬렉션의 내용을 표시하는 로직 추가
                content = self.db_manager.view_collection_content(selected_collection)
                st.write(f"'{selected_collection}'의 내용입니다.")
                st.markdown(content)
    
    def select_folder(self):
        root = Tk()
        root.withdraw()  # Tk 창 숨기기
        folder_path = filedialog.askdirectory()
        root.destroy()
        return folder_path
    
    def save_file_to_ragtmp_directory(file, filename):
        # ragtmp 디렉토리 경로 설정
        ragtmp_dir = os.path.join(os.getcwd(), 'ragtmp')

        # ragtmp 디렉토리가 없는 경우에만 생성
        if not os.path.exists(ragtmp_dir):
            os.makedirs(ragtmp_dir)

        # 파일을 ragtmp 디렉토리에 저장
        file_path = os.path.join(ragtmp_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

        # 디렉토리와 파일 이름을 결합하여 값 반환
        return file_path

    def delete_file(file_path):
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            st.warning("파일을 찾을 수 없습니다.")
            return False
        
        
    def setup_llmmodel_page(self):
        st.title("LLM 모델 챗봇")

        # LLM 소스 선택
        llm_source = st.radio("LLM 소스 선택:", ("LM Studio", "Ollama"))

        if llm_source == "Ollama":
            models = self.get_ollama_models()
            
            if not models:
                st.error("사용 가능한 Ollama 모델이 없습니다. 'ollama pull' 명령어로 모델을 다운로드 해주세요.")
                self.set_llm_model(lm_llm)
                return 

            selected_model = st.selectbox("사용할 Ollama 모델을 선택하세요:", models)
            st.write(f"선택한 모델: {selected_model}")

            # Ollama 모델 초기화
            llm = Ollama(model=selected_model)
            self.set_llm_model(llm)
        else:  # LM Studio
            models = self.get_lm_studio_models(BASE_URL)       
            
            if models:
                lm_llm.model_name = models
                self.set_llm_model(lm_llm)
                st.success("LM Studio 서버에 성공적으로 연결되었습니다.")
            else:
                self.set_llm_model(lm_llm)
                st.error("LM Studio 서버에 연결할 수 없습니다. 디폴트 셋팅을 따릅니다.")


    def setup_embed_page(self):
        st.header("임베딩")         
        collections = self.db_manager.list_collections()       
        selected_collection = st.selectbox("컬렉션을 선택하세요", collections, key="embed_collection_select")
        collection_name = selected_collection

        # 초기화 버튼
        if st.button("파일 업로더 초기화"):
            st.session_state.files_processed = False
            st.experimental_rerun()

        # 파일 처리 완료 플래그 확인
        if 'files_processed' not in st.session_state:
            st.session_state.files_processed = False

        # 파일 업로더를 조건부로 표시
        if not st.session_state.files_processed:
            uploaded_files = st.file_uploader("파일을 선택하세요", type=['.pdf','.pptx','.ppt', '.doc', '.docx','.hwp','.hwpx','.xlsx'], accept_multiple_files=True)
            
            if st.button("벡터저장", key="store_vector_collection"):            
                if uploaded_files is not None:                
                    total_chunks_stored = 0
                    for file in uploaded_files:
                        try:
                            # 텍스트 추출                
                            with st.spinner(f"{file.name} 파일의 텍스트 추출 중..."):
                                text = self.db_manager.extract_text_from_file(file, file.name)
                            
                            if not text:
                                st.warning(f"{file.name}에서 추출된 텍스트가 없습니다.")
                                continue

                            # 텍스트 처리 및 임베딩
                            with st.spinner(f"{file.name} 텍스트 처리 및 임베딩 중..."):
                                processed_text = text  # 필요한 경우 전처리 로직 추가
                                chunks_stored = self.db_manager.split_embed_docs_store(processed_text, file.name, collection_name)
                            
                            total_chunks_stored += chunks_stored
                            st.success(f"'{file.name}' 파일이 처리되어 {chunks_stored}개의 청크가 '{collection_name}' 컬렉션에 저장되었습니다.")
                        except Exception as e:
                            st.error(f"{file.name} 파일 처리 중 오류가 발생했습니다: {str(e)}")
                    
                    # 모든 파일 처리가 완료된 후
                    st.success(f"총 {total_chunks_stored}개의 청크가 저장되었습니다.")
                    st.session_state.files_processed = True
                    st.experimental_rerun()
        else:
            st.success("파일 처리가 완료되었습니다. 새 파일을 업로드하려면 '파일 업로더 초기화' 버튼을 클릭하세요.")
                
       

   
    def handle_user_input(self):
        with maincol1:
            user_question = st.chat_input("Enter your search query")
            if user_question:
                st.chat_message("user").markdown(user_question)
                st.session_state.messages.append(ChatMessage(role="user", content=user_question))
                st.session_state.last_query = user_question
                self.process_query(user_question)


    def process_query(self, query):
        keyword, parsed_query = self.parse_input(query)
        if (keyword == "f" or st.session_state.apply_filter) and st.session_state.rag_usage == "Rag+LLM":
            self.process_filtered_query(parsed_query)
        else:
            self.process_regular_query(parsed_query)

    def process_filtered_query(self, query):
        if st.session_state.selected_sources:
            docs = self.db_manager.get_documents_by_source(self.collection_name, st.session_state.selected_sources)
            if docs:
                self.generate_response(docs, query)
            else:
                st.write("No documents found for the selected sources.")
        else:
            st.info("검색할 문서를 선택하세요!")

    def process_regular_query(self, query):
        if st.session_state.RagUsage == "Rag+LLM":
            with st.spinner('Rag+LLM 답변 사전 준비중...'):
                docs = self.perform_search(query, self.db_manager, self.collection_name, st.session_state.selected_sources)                
            if docs:
                self.generate_response(docs, query)                
                self.display_search_results(docs)
            else:
                self.fallback_to_llm(query)
        else:
            self.generate_response(None, query)

    def generate_response(self, docs, query):
        try:
            chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True)
            with maincol1:
                with st.spinner('AI 답변을 기다리고 있는중...'):
                    response = chain.run(input_documents=docs or "", question=query)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append(ChatMessage(role="assistant", content=response))
        except Exception as e:
            st.error(f"LLM 서버연결 오류 발생: {e}")

    def fallback_to_llm(self, query):
        st.info("선택한 소스나 문서에서 관련 정보를 찾을 수 없어 LLM에 질의합니다.")
        self.generate_response(None, query)
     

    def run(self):
        self.setup_ui()
    
        # 사이드바 설정
        with st.sidebar:
            self.setup_sidebar()
        
        # 메인 영역
        self.handle_user_input()


if __name__ == '__main__':
    app = RAGChatApp()
    app.run()
