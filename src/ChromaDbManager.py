from dotenv import load_dotenv
from pathlib import Path
from tkinter import messagebox
import chromadb
from chromadb.config import Settings
import os
import uuid
import logging
from openpyxl import load_workbook
from pptx import Presentation
import json

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from ExtractTextFromFile import ExtractTextFromFile
from CustomSentenceTransformerEmbeddings import CustomSentenceTransformerEmbeddings as CSTFM

      
class ChromaDbManager:
    def __init__(self, persist_directory=""):
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        load_dotenv(dotenv_path=env_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(os.path.dirname(current_dir), "chroma_db")
        persist_directory =MODEL_PATH
        self.config_file = "chroma_config.json"
        self.persist_directory = persist_directory
        self.embeddings = CSTFM()
        #self.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        self.client = self._create_client()
        self.vectordb = None
        self.docnum = int(os.environ.get("DOC_NUM"))
        self.chunk_size = int(os.environ.get("CHUNK_SIZE"))
        self.chunk_overlap=int(os.environ.get("CHUNK_OVERLAP"))
        self.extractor = ExtractTextFromFile()
         

    def _create_client(self):        
        return chromadb.PersistentClient(path=self.persist_directory)
    
    def set_return_docnum(self, docnum):
        self.docnum = docnum
       
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def get_persist_directory(self):
        return self.persist_directory

    def save_config(self):
        config = {'persist_directory': self.persist_directory}
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

    def get_or_create_collection(self, collection_name):
        if self.vectordb is None or self.vectordb._collection.name != collection_name:
            self.vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,               
            )
        return self.vectordb
    
    
    def extract_text_from_file(self, file, file_name):
        return self.extractor.extract_text_from_file(file, file_name)
    
    def set_persist_directory(self, new_directory):
        if self.persist_directory != new_directory:
            self.persist_directory = new_directory
            # 기존 클라이언트를 닫고 새로운 클라이언트를 생성
            if hasattr(self, 'client'):
                self.client.close()
            self.client = self._create_client()
            
    def create_collection(self, collection_name):       
        try:
            self.client.create_collection(name=collection_name)
            #print(f"create collection info: {self.client.get_collection(name=collection_name).count()}")
            return f"Collection '{collection_name}' created successfully."
        except ValueError as e:
            return f"Error creating collection: {e}"

    def delete_collection(self, collection_name):
        try:
            self.client.delete_collection(name=collection_name)
            #print(f"delete collection info: {self.client.get_collection(name=collection_name).count()}")
            return f"Collection '{collection_name}' deleted successfully."
        except ValueError as e:
            return f"Error deleting collection: {e}"

    def list_collections(self):
        collections = self.client.list_collections()
        return [collection.name for collection in collections]
    
    def view_collection_content(self, collection_name, docnum=20):
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # 컬렉션의 총 개수 확인
            total_count = collection.count()
            #print(f"Total items in collection: {total_count}")

            # 한 번에 모든 데이터 가져오기, 서버 측에서 limit 적용
            items = collection.get(limit=docnum)
            
            if not items['ids']:
                return f"Collection '{collection_name}' is empty or data could not be retrieved."

            content = []
            for i in range(len(items['ids'])):
                item_info = {
                    "ID": items['ids'][i],
                    "Metadata": items['metadatas'][i] if items['metadatas'] else "No metadata",
                    "Document": items['documents'][i] if items['documents'] else "No document"
                }
                content.append(item_info)

            return content

        except ValueError as e:
            return f"Error viewing collection: {e}"
        except Exception as e:
            return f"Unexpected error occurred: {e}"

    


    def get_subdirectories(path):
        try:
            return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        except PermissionError:
            messagebox.showwarning("Permission Denied", f"Permission denied to access {path}")
        return []
    
   
   
    def verify_storage(self, collection_name):
        collection = self.client.get_collection(name=collection_name)
        count = collection.count()
        #print(f"Verification: Collection '{collection_name}' has {count} items.")
        if count > 0:
            sample = collection.peek()
            #print(f"Sample item: {sample}")
        return count
    
    def test_embedding():
        #embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        #embeddings = OllamaEmbeddings(model="llama3:instruct")
        #embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        embeddings = CSTFM()
        test_text = "This is a test sentence."
        #print(test_text)
        result = embeddings.embed_query(test_text)
        #print(f"Embedding dimension: {len(result)}")
        #print(f"First few values: {result[:5]}")

    
    def store_in_chroma(self, text, filename, collection_name, progress_callback=None):
        collection = self.client.get_or_create_collection(collection_name)
        
        existing_docs = self.get_ids_by_source(collection_name=collection_name, source=filename)
        if existing_docs[0]:
            messagebox.showinfo("Success", f"문서 '{filename}'이(가) 이미 존재합니다. 업데이트합니다.")
            chunks = self.chunk_text(text,self.chunk_size)
            total_chunks = len(chunks)                    
            for i, chunk in enumerate(chunks):
                collection.update(
                    documents=[chunk],
                    metadatas=[{"source": filename}],
                    ids=existing_docs[0]
                )
                if progress_callback:
                    progress_callback((i + 1) / total_chunks * 100)
        
        else:
            messagebox.showinfo("Success", f"새 문서 '{filename}'을(를) 추가합니다.")
            chunks = self.chunk_text(text, self.chunk_size)
            total_chunks = len(chunks)
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": filename}],
                    ids=[ids[i]]
                )
                if progress_callback:
                    progress_callback((i + 1) / total_chunks * 100)       

        return total_chunks
    

   
    def get_list_collections(self):
        return [col.name for col in self.client.list_collections()]
    
    def split_embed_docs_store(self, text, file_name, collection_name):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting document processing")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(self.chunk_size), chunk_overlap=int(self.chunk_overlap))
        
        chunks = text_splitter.split_documents(documents=text)
        logger.info(f"Created {len(chunks)} chunks")
        
        try:
            logger.info("Initializing embedding model")
            embeddings = self.embeddings
            logger.info("Creating Chroma vector store")
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=f"{self.persist_directory}/{collection_name}",
                client=self.client,
                collection_name=collection_name,
                client_settings=
                {"anonymized_telemetry":False}
            )
            
            logger.info("Persisting vector store")
            vectordb.persist()
            
            logger.info("Verifying storage")
            count = self.verify_storage(collection_name)
            
            return count
        except Exception as e:
            logger.error(f"Error in split_embed_docs_store: {e}", exc_info=True)
            raise


    def search_collection(self, collection_name, query, n_results, inscore=350):
        vectordb = self.get_or_create_collection(collection_name)
        
        # query_embedding은 이제 필요 없습니다.
        # query_embedding = self.embeddings.embed_query(query)
        
        self.set_return_docnum(n_results)
        
        results = vectordb.similarity_search_with_score(query, k=n_results)
        
        formatted_results = []
        for doc, score in results:
            if score <= inscore:
                formatted_results.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
        
        return formatted_results
    

    def get_all_documents_source(self, collection_name, source_search):
        collection = self.client.get_collection(collection_name)
        all_docs = collection.get()
        
        if source_search is None or len(source_search) == 0:
            # source_search가 비어있으면 모든 결과 반환  100개로 제한
            all_sources = list(dict.fromkeys(metadata.get('source', 'Unknown') for metadata in all_docs['metadatas']))[:100]  # 최대 50개로 제한
            filtered_sources = []
            for s in all_sources:
                filtered_sources.append(s)
            # print(filtered_sources)   
            return filtered_sources
        else:
            all_sources = list(dict.fromkeys(metadata.get('source', 'Unknown') for metadata in all_docs['metadatas']))
            filtered_sources = []
            for s in all_sources:
                if source_search.lower() in s.lower():
                    if not any(fs.lower() == s.lower() for fs in filtered_sources):
                        filtered_sources.append(s)       
            return filtered_sources
    
    def get_documents_by_source(self, collection_name, sources):
        collection = self.client.get_collection(collection_name)
        
        #print(f"Searching for documents with sources: {sources}")
        
        if isinstance(sources, str):
            sources = [sources]  # 단일 문자열을 리스트로 변환
        
        all_results = []
        
        for source in sources:
            results = collection.get(
                where={"source": source}
            )
    
            for doc, metadata in zip(results['documents'], results['metadatas']):
                all_results.append(Document(page_content=doc, metadata=metadata))
        
        #print(f"Found {len(all_results)} documents in total")
        
        return all_results
    
    def get_ids_by_source(self, collection_name, source):
        collection = self.client.get_collection(collection_name)
        
        #print(f"Searching for document IDs with source '{source}'")
        
        results = collection.query(
            query_texts=[""],  # 빈 쿼리 텍스트
            where={"source": source},
            include=["ids"]
        )

        matching_ids = results['ids'][0]  # 'ids'는 리스트의 리스트 형태로 반환됩니다
        
        if matching_ids:
            print(f"Found {len(matching_ids)} document(s) with source '{source}'")
        else:
            print(f"No documents with source '{source}' found in the collection")
        
        return matching_ids
    
    
    def check_source_exists(self, collection_name, source):
        collection = self.client.get_collection(collection_name)
        
        # UploadedFile 객체인 경우 파일 이름을 사용
        if hasattr(source, 'name'):
            source = source.name
        
        #print(f"Checking if documents with source '{source}' exist")
        extension = source.split('.')[-1]
        extension = extension.lower()
        if extension == "hwp": 
            #hwp를 내부적으로 pdf로 저장하기 때문에 확장자를 변경하여 비교
            source = source.rsplit('.', 1)[0]+".pdf"        
        
        try:
            results = collection.get(
                where={"source": str(source)},
                include=["metadatas"]
            )

            # metadatas가 비어있지 않으면 문서가 존재한다고 판단
            exists = len(results['metadatas']) > 0
            
            if exists:
                print(f"Documents with source '{source}' exist in the collection")
            else:
                print(f"No documents with source '{source}' found in the collection")
            
            return exists
        except Exception as e:
            print(f"Error querying the database: {str(e)}")
            return False