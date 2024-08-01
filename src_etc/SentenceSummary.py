import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest

def summarize_text(text, num_sentences=2):
    # NLTK 데이터 다운로드 (처음 실행할 때만 필요함)
    nltk.download('punkt')
    nltk.download('stopwords')

    # 문장 토큰화
    sentences = sent_tokenize(text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    word_frequencies = FreqDist(word for word in words if word not in stop_words)

    # 문장 점수 계산
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    # 상위 n개의 문장 선택
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # 요약 생성
    summary = ' '.join(summary_sentences)
    return summary


# 사용 예시
text = """
자연어 처리(NLP)는 컴퓨터 과학과 인공지능의 한 분야로, 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 기술입니다.
이 기술은 기계 번역, 감정 분석, 텍스트 요약, 챗봇 등 다양한 응용 분야에서 사용됩니다.
NLP는 언어학, 컴퓨터 과학, 수학, 통계학 등 여러 학문의 융합으로 발전해 왔으며, 최근 딥러닝 기술의 발전으로 더욱 정교해지고 있습니다.
자연어 처리 기술은 우리의 일상 생활에서도 널리 사용되고 있으며, 스마트폰의 음성 인식, 검색 엔진의 질의 응답 시스템, 소셜 미디어의 콘텐츠 분석 등에 활용되고 있습니다.
"""

summary = summarize_text(text,1)
print(summary)
