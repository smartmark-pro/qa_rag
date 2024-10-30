import streamlit as st
import time
import requests
import json
import pandas as pd
from openai import OpenAI, Stream

st.title("中文问答匹配")
st.header('基于开源模型bge的简易RAG')

# 模型相关接口参见 run_rag_service.py
# 本应用参考https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

@st.cache_data
def create_embeddings(df: pd.DataFrame, embeddings_id: str):
    from langchain.document_loaders import DataFrameLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    df["text"] = df.apply(lambda x: x["问题"]+"[QA]"+x["答案"], axis=1)
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    print(len(documents))
    # 简单使用控制问题答案长度， 避免embeding超过长度
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    
    sentences = [t.page_content for t in texts]
    origin_qa = df[['问题', '答案']].to_dict("records")
    origin_qa = {i: qa for i, qa in enumerate(origin_qa)}
    if texts:
        url = st.secrets.remote.get("algo_url")
        res = requests.post(url=url+"/v1/embeddings", data=json.dumps({
            "model_name": "", 
            "sentences": sentences,
            "embeddings_id": embeddings_id,
        }))
        try:
            if res.status_code==200:
                r = res.json()
            else:
                st.error(res)
        except Exception as e:
            st.exception(e)
    return origin_qa


st.subheader("上传问答对")
file = st.file_uploader("excel文件, 注意字段必须包含'问题'和'答案'")
if not file:
    st.subheader("上传样例数据")
    example_data = pd.DataFrame({
        '问题': ["测试问题1", "测试问题2"],
        '答案': ["测试答案1", "测试答案2"]
    })
    st.dataframe(example_data)
    df = example_data
else:
    st.subheader("已上传数据")
    try:
        df = pd.read_excel(file)
        st.dataframe(df[:3])
    except Exception as e:
        st.exception(e)
    if len(df)>1000:
        st.error("演示项目，建议数据不超过1000条，否则预处理时间过长导致服务阻塞")
        # 重置未为空数据
        df = pd.DataFrame({
            '问题': [],
            '答案': []
        })

import hashlib


def generate_dataframe_id(df):
    # 将 DataFrame 转换为字符串
    df_string = df.to_string(index=False)
    # 计算哈希值
    return hashlib.sha256(df_string.encode()).hexdigest()
embeddings_id = generate_dataframe_id(df)
# 直接streamlit里面生成唯一id
print(embeddings_id)
origin_qa = create_embeddings(df, embeddings_id)
print(str(origin_qa)[:100])
embeddings_status = {}


st.sidebar.header('可调整参数')
# welcome = st.sidebar.text_input(label="默认欢迎语", key="默认欢迎语")
match_method = st.sidebar.selectbox("问答对召回方式", ("sparse", "dense", "colbert", "custom"))
topk = st.sidebar.slider('最大召回数量', 1, 100, 10)
if match_method == "sparse":
    default_value = 0
else:
    default_value = 0.5
min_score = st.sidebar.number_input("召回最小阈值", min_value=default_value)

custom_weights = []
if match_method=="custom":
    ratio1 = st.number_input("sparse比例值", value=1, min_value=0.0, max_value=1.0, step=0.1)
    ratio2 = st.number_input("dense比例值", value=1, min_value=0.0, max_value=1.0, step=0.1)
    ratio3 = st.number_input("colbert比例值", value=1,  min_value=0.0, max_value=1.0, step=0.1)
    custom_weights = [ratio1, ratio2, ratio3]

is_use_rerank = st.sidebar.checkbox("是否使用rerank召回")
if is_use_rerank:
    rerank_min_score = st.sidebar.number_input("重排序最小阈值", min_value=0.0)
    rerank_topk = st.sidebar.slider('重排序最大召回数量',1, 5, 1)
is_use_big_model = st.sidebar.checkbox("是否使用大模型回答")

no_answer = st.sidebar.text_input(label="默认答案", value="该问题答案不存在， 请联系人工客服")


if is_use_big_model:
    openai_model_name = st.sidebar.text_input("大模型", value="gpt-3.5-turbo")
    custom_prompt = st.sidebar.text_area(label="默认提示", value="你是一名专业的人工客服， 仅根据问答库:{context}， 回答用户咨询的问题:{query}， 回答时，注意不要回复无关内容，如果没有答案请回答{no_answer}", height=100)
    if "{context}" not in custom_prompt or "query" not in custom_prompt:
        st.error("自定义提示格式错误， 请参考默认例子")
    llm_token = st.sidebar.text_input(label="openai token", value="")
    if llm_token:
        api_key = llm_token
        base_url = "https://api.openai.com/v1"
    else:
        # 暂时使用该免费api https://github.com/popjane/free_chatgpt_api
        api_key = st.secrets.remote["openai_api_key"]
        base_url = st.secrets.remote["openai_url"]
    client = OpenAI(api_key=api_key, base_url=base_url)
# TODO 测评
# test_method = st.sidebar.selectbox(
#      '简单评测方法',
#      ('构造方法1', '构造方法2'))



def get_embeddings_score(query, method, topk, weights):
    url = st.secrets.remote.get("algo_url")
    res = requests.post(url=url+"/v1/embeddings/score", data=json.dumps({"query": [query], "method": method, "topk": topk, "weight": weights, "embeddings_id": embeddings_id, "min_score":min_score}))
    r = None
    try:
        if res.status_code==200:
            r = res.json()
        else:
            st.error(res.json())
    except Exception as e:
        st.exception(e)
        return None
    return r


def get_rerank_score(sentences, topk, min_score):
    # fixme
    url = st.secrets.remote.get("algo_url")
    res = requests.post(url=url+"/v1/rerank", data=json.dumps({"sentences": sentences, "min_score": min_score, "topk": topk}))
    try:
        if res.status_code==200:
            r = res.json()
        else:
            st.error(res.json())
    except Exception as e:
        st.exception(e)
        return None
    return r



def get_answer(score_info):
    if score_info:
        if is_use_big_model:
            last_m = st.session_state.messages[-1]
            qa = [str(origin_qa.get(sc[0])) for sc in score_info]
            params = {
                "context": "\n".join(qa),
                "query": last_m["content"],
                "no_answer": no_answer
            }
            response = client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ] + [{"role": last_m["role"], "content": custom_prompt.format(**params)}],
                stream=True,
            )
        else:
            # 否则取第一个作为答案
            response = origin_qa.get(score_info[0][0]).get("答案")
    else:
        response = no_answer
    return response


# Streamed response emulator
def response_generator(query):
    
    score_info= get_embeddings_score(query, method=match_method, topk=topk, weights=custom_weights)
    print("m3", score_info)
    response = None
    if not score_info:
        response = no_answer
    else:
        if is_use_rerank:
            # 调用rerank接口
            sentences = [(query, origin_qa.get(sc[0]).get("问题")+"?"+origin_qa.get(sc[0]).get("答案")) for sc in score_info]
            score_info = get_rerank_score(sentences, topk=rerank_topk, min_score=rerank_min_score)
            print("rerank", score_info)
        response = get_answer(score_info)
        print("final", response)
        if not response:
            response = "内部错误"
    if isinstance(response, Stream):
        for w in response:
            yield w
    else:
        for word in response.split():
            yield word
            time.sleep(0.01)



st.subheader("对话测试")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("输入问题"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
