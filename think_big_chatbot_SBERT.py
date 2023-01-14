#Thinkbig_chatbot SBERT(sentence_transformers)
import streamlit as st
from streamlit_chat import message
# streamlit_chat은 python 3.8버전 이상에서만 제대로 동작
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import Literal, Optional, Union
import streamlit.components.v1 as components

@st.cache(allow_output_mutation=True)
# cache : 모델을 여러번 부르지 않고 한번만 불러오는 역할
# Streamlit의 캐시 주석으로 함수를 표시하면 함수가 호출될 때마다 다음 세 가지를 확인해야 한다고 Streamlit에 알린다.
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model
# 모델은 미리 학습된 SentenceTransformer을 가져와서 사용
# SentenceTransformer는 최신 문장, 텍스트 및 이미지 임베딩을 위한 python 프레임워크
# 이 프레임워크를 사용하여 100개 이상의 언어에 대한 문장/텍스트 임베딩을 계산
# 그런 다음 이러한 임베딩을 예를 들어 코사인 유사도와 비교하여 유사한 의미를 가진 문장을 찾는다.


# wellness_dataset_final.csv : ai허브의 감성대화 데이터와 챗봇 데이터를 병합한 후 전처리한 데이터
@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset_final.csv')  # 임베딩된 데이터셋 로드
    df['embedding'] = df['embedding'].apply(json.loads)  # 임베딩
    return df

# streamlit에 위에서 정의한 함수를 사용하여 모델과 데이터셋 로드
model = cached_model()
df = get_dataset()

# 화면에 표시되는 순서대로 출력
st.title('자연어처리 프로젝트')  # 제목
st.header('심리상담 챗봇')       # 헤더
# st.subheader("서브헤더")
# st.text("텍스트")
st.markdown("❤️chatbot_think_big")         # 마크다운
#st.subheader("")
st.markdown("""
    🙂 자연어처리 1차 팀프로젝트를 위한 심리 상담 챗봇입니다.
    """
    """
    💜 SentenceTransformer를 사용하여 문장을 임베딩하고 이를 코사인 유사도와 함께 비교하여 가장 유사한 답변을 채택합니다.
    """)

# 왼쪽 sidebar 부분
st.sidebar.header("NLP PROJECT")
st.sidebar.subheader("TEAM : Think_Big")
st.sidebar.subheader("팀원")
st.sidebar.text("조인환(팀장)")
st.sidebar.text("김영진")
st.sidebar.text("최예은")
st.sidebar.text("백서윤")

# session_state : 각 사용자 세션에 대해 재실행 간에 변수를 공유하고 상태를 저장하고 유지
# 챗봇이 대화한 내용을 저장하는 generated session_state를 만든다
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# 유저가 대화한 내용을 저장하는 past session_state를 만든다
if 'past' not in st.session_state:
    st.session_state['past'] = []
## session_state를 사용하면 streamlit이 자동으로 재실행되도 초기화가 되지 않도록 한다.

# form을 만들어서 유저의 입력박스와 전송 버튼을 만든다
with st.form('form', clear_on_submit=True):  
    # clear_on_submit=True : 전송 버튼을 누르면 텍스트 박스가 자동으로 지워진다.
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

# 유저의 input에 질문이 입력되면 인코딩하여 벡터화한다
if submitted and user_input:
    embedding = model.encode(user_input)

    # 유저의 질문과 데이터의 질문들을 코사인 유사도로 비교한 후,
    # 가장 유사도가 높은 답변을 출력
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
 
    # 유저의 질문을 past에 저장
    st.session_state.past.append(user_input)
    # 챗봇의 답변을 generated에 저장
    st.session_state.generated.append(answer['A'])

# 유저의 질문과 챗봇의 답변을 메세지창에 출력하도록 하는 구문
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
