import time
from tempfile import template

import requests

from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import uvicorn
from fastapi import FastAPI, Request
from collections import defaultdict, deque

TARGET_GROUP = "组id"
TARGET_USER = "用户id(数字)"

SEND_URL = ""

OCCUPANCY_LIMIT = 0.2  # 发言总占比阈值 (20%)
TIME_WINDOW_SHORT = 300  # 时间窗口，单位：秒 (5分钟)
MESSAGE_LIMIT = 10  # 单位时间内的发言次数限制
app = FastAPI()

user_messages_count = defaultdict(int)  # 每个用户的发言次数
total_messages = 0  # 记录总发言次数
message_times = deque()  # 记录所有发言时间

user_messages_time = deque()  # 记录用户的发言时间
user_messages_content = deque() # 记录用户的发言内容

QIANFAN_AK = "API"
QIANFAN_SK = "API"

template = """
你是一位吐槽方面的专家，你的任务是根据用户过去的发言来对用户的水群行为进行非常无礼的吐槽。

请注意：你的吐槽内容必须保持在20字以内

用户发言列表：
{text}

吐槽内容：
"""

prompt = PromptTemplate.from_template(template.strip('\n'))
prompt.format(
    text = "text"
)

chat = QianfanChatEndpoint(
    qianfan_ak = QIANFAN_AK,
    qianfan_sk = QIANFAN_SK,
    model = "ERNIE-Speed-8K",
    temperature = 0.9
)

chain = (
    prompt | chat | StrOutputParser()
)

def format_texts(texts: deque):
    rs = ""
    idx = 1
    for txt in texts:
        rs += f"发言{idx}: {txt}\n"
        idx += 1
    return rs

def llm_warn(texts):
    res = chain.invoke({
        "text": format_texts(texts).strip('\n')
    })
    send_warning(res.strip('\n'))

def send_warning(text):
    warn_txt = f"[龙王警告⚠️]  {text}"
    requests.post(SEND_URL, json={
        'group_id': TARGET_GROUP,
        'message': [{
            'type': 'text',
            'data': {
                'text': warn_txt
            }
        }]
    })

def record_message_total(user_id, data):
    global total_messages, user_messages_content

    current_time = time.time()

    if data["message"][0]["type"] == "text":
        msg = data["message"][0]['data']['text']
    else:
        msg = "[表情包]"
    if msg.startswith("[龙王警告⚠️]"):
        return

    # 记录当前用户的发言
    user_messages_count[user_id] += 1

    total_messages += 1
    message_times.append((user_id, current_time))

    # 检查用户的发言占比
    if user_id == TARGET_USER and total_messages > 100 and (user_messages_count[user_id] / total_messages) > OCCUPANCY_LIMIT:
        llm_warn(user_messages_content)


def record_message(user_id, data):
    if user_id != TARGET_USER:
        return
    global user_messages_time, user_messages_content

    current_time = time.time()

    # 记录当前用户的发言时间
    user_messages_time.append(current_time)
    if data["message"][0]["type"] == "text":
        msg = data["message"][0]['data']['text']
    else:
        msg = "[表情包]"
    if msg.startswith("[龙王警告⚠️]"):
        return
    user_messages_content.append(msg)

    # 清理超过时间窗口的发言记录
    while user_messages_time and user_messages_time[0] < current_time - TIME_WINDOW_SHORT:
        user_messages_time.popleft()
        user_messages_content.popleft()

    # 检查发言次数是否超过限制
    if len(user_messages_time) > MESSAGE_LIMIT:
        llm_warn(user_messages_content)

@app.post("/")
async def root(request: Request):
    data = await request.json()  # 获取事件数据
    # 拦截群的消息
    if data["message_type"] == "group" and data["group_id"] == TARGET_GROUP:
        user_id = data["user_id"]
        record_message_total(user_id, data)
        record_message(user_id, data)
    return {}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)