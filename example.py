# Libraries
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from utils.llm import claude_v2

# Global Variables
theme_plotly = None # None or streamlit
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Config
st.set_page_config(page_title='Data Cleaning', page_icon=':bar_chart:', layout='wide')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


# 其他必要的库
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def clean_ocr_results(text):
    sep = [i['text']+'\n' for i in eval(text)['result']['lines']]
    return '\n'.join(sep)

def recognize(img):
    # 通用文字识别
    url = 'https://api.textin.com/ai/service/v2/recognize'
    head = {}
    try:
        # image = get_file_content(self._img_path)
        head['x-ti-app-id'] = st.session_state['ocr_app_id']
        head['x-ti-secret-code'] = st.session_state['ocr_secret_code']
        result = requests.post(url, data=img, headers=head)
        return result.text
    except Exception as e:
        return e

def fetch_and_process_OCR(img1, img2):
    # 使用API KEY进行OCR识别的逻辑，这里仅为伪代码
    text1 = recognize(img1)
    text2 = recognize(img2)
    return text1, text2

def sort_text(text):
    # 调用大语言模型API进行处理，这里为伪代码
    prompt = '''
Human:    
以下为OCR识别的食品包装上的文字：
```
%s
```

我们需要对以上信息进行整理，请按照将提供的内容整理为一段markdown，其中要包含：配料、保质期、贮存方式、使用方式、产品标准号、食品生产许可证号、生产日期、委托商、委托商地址、制造商、制造商地址等14项信息。

请去除OCR结果中看起来无关的信息。
请将结果输出为json格式，如遇到某项信息缺失，请填”N/A”。json示例如下：
```json
{"配料"："水，有机豆瓣酱（水、有机大豆、有机小麦粉、食用盐），有机葵花籽油，有机干香菇，有机白砂糖，香辛料", etc...}
```
Assistant:''' % text
    payload = {
        "prompt": prompt,
        "max_tokens_to_sample": 1000,
        "temperature": 0.1,
        "top_p": 0.9, 
    }
    response = claude_v2(payload)
    print(response)
    # parse response, get text between ```json and ```
    response = response.split('```json')[1].split('```')[0]
    response = eval(response)
    return response


def compare_texts(text1, text2):
    # 比较新旧标签文本
    prompt = '''Human:
你是一个食品包装检查员，任务是检查某食品品牌在更新包装后，文字是否有改变，你接到一项新的任务，需要检查新旧包装文字的差异，并输出结果：
旧标签：
```
%s
```
新标签：
```
%s
```
请为以上所有新旧包装文本的提供完整的对比报告，请只输出新旧有所差别的部分，相同的部分请忽略。请勿输出json结果之外的任何内容。
格式上请参考Json：```json{"配料": "不相同，配料一栏中，新标签在香辛料中多了（八角粉）", "保质期": "相同", "贮存方式": "相同", "使用方式": ...}```

Assistant:
''' % (text1, text2)
    payload = {
        "prompt": prompt,
        "max_tokens_to_sample": 1000,
        "temperature": 0.1,
        "top_p": 0.9, 
    }
    response = claude_v2(payload)
    print(response)
    # parse response, get text between ```json and ```
    # response = response.split('```json')[1].split('```')[0]
    # response = eval(response)
    return response

def main():
    st.title('AI食品包装检查')

    # Step 1: 上传旧版包装图片和新版包装图片
    uploaded_file_1 = st.file_uploader("上传旧版包装图片", type=['jpg', 'png', 'jpeg'])
    uploaded_file_2 = st.file_uploader("上传新版包装图片", type=['jpg', 'png', 'jpeg'])

    st.session_state['ocr_app_id'] = st.text_input("输入OCR API Key:", value='29d793cc237c56958273df6b9db40f68')
    st.session_state['ocr_secret_code'] = st.text_input("输入OCR Secret Code:", value='12fdf3df14ebef3bfce0e26c2171474f')
    
    if uploaded_file_1 and uploaded_file_2 and st.session_state['ocr_app_id'] and st.session_state['ocr_secret_code']:
        if st.button('识别', use_container_width=True):
            st.subheader('第一步：归档后文字')
            text1, text2 = fetch_and_process_OCR(uploaded_file_1, uploaded_file_2)
            
            cleaned1 = clean_ocr_results(text1)
            cleaned2 = clean_ocr_results(text2)
            text_columns = st.columns(2)
            text_columns[0].subheader('旧版包装文字')
            text_columns[0].write(cleaned1)
            text_columns[1].subheader('新版包装文字')
            text_columns[1].write(cleaned2)
            
            # Step 2: 选择需要比较的文字
            st.subheader('第二步：归档后文字')
            # add loading sign
            with st.spinner('正在处理中...'):
                response1 = sort_text(cleaned1)
                response2 = sort_text(cleaned2)

            # Create columns for the predefined search samples
            sorted_columns = st.columns(2)
            sorted_columns[0].subheader('旧版包装文字')
            sorted_columns[0].write(response1)
            sorted_columns[1].subheader('新版包装文字')
            sorted_columns[1].write(response2)

            # Step 3: 对比新旧文字
            st.subheader('第三步：对比新旧文字')
            with st.spinner('正在处理中...'):
                comparison_result = compare_texts(response1, response2)
                st.write(comparison_result)  # 显示比较结果
            # 这里你可能需要对response进一步的处理，以确保它们可以正常传递给compare_texts函数

            # if st.button('对比'):
            #     comparison = compare_texts(response1, response2)
            #     st.json(comparison)  # 显示比较结果

if __name__ == "__main__":
    main()
