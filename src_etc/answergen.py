import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 샘플 질문-답변 쌍 (실제 사용시 더 많은 데이터 필요)
qa_pairs = [
    ("인공지능이란 무엇인가요?", "인공지능은 인간의 학습능력, 추론능력, 지각능력 등을 컴퓨터 프로그램으로 실현한 기술입니다."),
    ("머신러닝의 정의는?", "머신러닝은 컴퓨터가 명시적인 프로그래밍 없이 데이터로부터 학습하여 성능을 향상시키는 인공지능의 한 분야입니다."),
    ("딥러닝이란?", "딥러닝은 여러 층의 인공신경망을 사용하여 데이터로부터 복잡한 패턴을 학습하는 머신러닝의 한 기법입니다."),
    ("자연어 처리란 무엇인가요?", "자연어 처리는 컴퓨터가 인간의 언어를 이해, 해석, 생성할 수 있게 하는 인공지능의 한 분야입니다."),
    ("강화학습에 대해 설명해주세요.", "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습하는 머신러닝의 한 종류입니다.")
]

# TF-IDF 벡터라이저 초기화
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform([pair[0] for pair in qa_pairs])

def preprocess_text(text):
    # 토큰화 및 불용어 제거
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words and token not in string.punctuation]

def find_similar_questions(user_question, top_n=3):
    # 사용자 질문을 벡터화
    user_vector = vectorizer.transform([user_question])
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    
    # 상위 N개의 유사한 질문 인덱스 찾기
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return [qa_pairs[i] for i in top_indices]

def generate_follow_up_questions(question, answer, num_questions=5):
    # 질문과 답변에서 중요 단어 추출
    question_tokens = set(preprocess_text(question))
    answer_tokens = set(preprocess_text(answer))
    important_words = list(question_tokens.union(answer_tokens))
    
    # 질문 템플릿
    templates = [
        "{}에 대해 더 자세히 설명해주세요.",
        "{}의 장점은 무엇인가요?",
        "{}와 관련된 예시를 들어주실 수 있나요?",
        "{}가 실생활에서 어떻게 적용되나요?",
        "{}의 미래 전망은 어떤가요?"
    ]
    
    # 예상 질문 생성
    follow_up_questions = []
    for _ in range(num_questions):
        template = random.choice(templates)
        word = random.choice(important_words)
        follow_up_questions.append(template.format(word))
    
    return follow_up_questions

# Streamlit 앱
st.title("ML-based Q&A with Follow-up Questions")

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 사용자 입력
user_question = st.text_input("질문을 입력하세요:")

if user_question:
    # 유사한 질문-답변 쌍 찾기
    similar_qa = find_similar_questions(user_question)[0]
    answer = similar_qa[1]
    
    # 채팅 기록에 추가
    st.session_state.chat_history.append(("User", user_question))
    st.session_state.chat_history.append(("AI", answer))
    
    # 예상 질문 생성
    follow_up_questions = generate_follow_up_questions(user_question, answer)

    # 채팅 기록 표시
    for role, message in st.session_state.chat_history:
        st.write(f"**{role}:** {message}")

    # 예상 질문 버튼 표시
    st.write("**예상 질문:**")
    for i, question in enumerate(follow_up_questions):
        if st.button(question.strip(), key=f"question_{i}"):
            # 선택된 예상 질문에 대한 답변 생성
            new_similar_qa = find_similar_questions(question)[0]
            new_answer = new_similar_qa[1]
            
            # 채팅 기록에 추가
            st.session_state.chat_history.append(("User", question))
            st.session_state.chat_history.append(("AI", new_answer))
            
            # 페이지 새로고침
            st.rerun()

# 채팅 초기화 버튼
if st.button("대화 초기화"):
    st.session_state.chat_history = []
    st.rerun()