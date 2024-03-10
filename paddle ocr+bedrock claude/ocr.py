import os
import random
from flask import Flask, render_template , send_from_directory, request , jsonify
import paddleocr
from paddleocr import PaddleOCR, draw_ocr
import boto3
import json
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__,template_folder="templates")

@app.route('/dataset/<filename>')
def dataset(filename):
    return send_from_directory('dataset', filename)


@app.route('/')  
def index():
    # 随机选择图片
    folder_path = './dataset'
    file_names = os.listdir(folder_path)
    jpg_files = [file for file in file_names if file.endswith('.jpg')]
    random_file = random.choice(jpg_files)
    print (random_file)
    return render_template('index.html', image_file=random_file)

@app.route('/<image_name>')  
def process_image(image_name):
    # 检查图片文件是否存在
    file_path = os.path.join('dataset', f'{image_name}.jpg')
    if not os.path.isfile(file_path):
        return "图片不存在", 404

    return render_template('index.html', image_file=f'{image_name}.jpg')

@app.route('/get_ocr_results')
def get_ocr_results():
    file_name = request.args.get('file_name', '')
    img_path = os.path.join('dataset', file_name)


    # OCR识别
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                        # det_model_dir="/home/ubuntu/model/ch_PP-OCRv4_det_server_infer",
                        # rec_model_dir="/home/ubuntu/model/ch_PP-OCRv4_rec_server_infer")  
    ocr_result = ocr.ocr(img_path, cls=True)


    # 将处理后的图片转换为 Base64 编码
    result = ocr_result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/ubuntu/simfang.ttf')  # 确保指定正确的字体路径
    im_show = Image.fromarray(im_show)        
    buffered = BytesIO()
    im_show.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return jsonify({
        'image_base64': img_str,
        'ocr_result':ocr_result
    })

@app.route('/get_claude_results',methods=['POST'])
def get_claude_results():
    data = request.get_json() # 获取 JSON 数据
    ocr_result = data.get('ocr_result')    

    # 调用Claude生成描述
    ocr_new_result = remove_confidence(ocr_result) # 去掉置信度
    
    prompt = generate_prompt(ocr_new_result)

    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2',endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com')
    modelId = 'anthropic.claude-v2'

    response = call_claude(prompt, modelId, bedrock)
    completion = get_completion(response)

    print (completion)
    # 展示结果
    return jsonify({
        'captions': completion
    })

def remove_confidence(data_list):
    new_list = []
    
    # Helper recursive function to delve into the nested lists
    def process_item(item):
        # Check if this is the list containing the tuple with confidence
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], tuple):
            coordinates = item[0]
            text = item[1][0]  # Only take the text part, discard the confidence score
            return [coordinates, text]
        elif isinstance(item, list):
            return [process_item(subitem) for subitem in item]
        else:
            return item
    
    # Process each item using the helper function
    for item in data_list:
        new_list.append(process_item(item))
    
    return new_list

def generate_prompt(ocr_result):
    prompt = """\n\nHuman:
    任务描述：
    1.解析输入的列表,这个列表是一个OCR程序识别小票后返回的结果。列表中的每个item包含了文字在小票中的相对位置和文字。识别文本所在位置和文字之间的相对关系。
    2.规格化识别的文本框以及文字结果：将可能断行的中文文字，根据文本框的相对位置，拼接成完整的句子。将可能断行的日期，根据文本框的相对位置，拼接为完整的日期。
    3.识别关键字段，包括:"门店名称", "门店地址", "日期", "流水号", "应收金额", "实收金额", "优惠金额", "会员卡号", "药品信息"。

    识别的规则:
    <rule>
    1.位于最上方第一行或者第二行的文本框中的文字，代表内容是“门店名称”，不可能包含"再打印"相关文字。
    2.日期一般是以"日期:"开头。并且遵循YYYYMMDD HH:MM:SS的格式,DD和HH中间有一个空格。
    3."流水号", "应收", "实收", "优惠", "会员卡号",这几个字段对应的值为纯数字,没有汉字或其他类别字符。
    4.实收总额和收款总额不同，收款总额减去找零即为实收总额。
    5.药品信息可能有多条，包含多种药品，都包含在药品信息这个字段中,多条药品之间分为多行输出，并且分别提取不同药品的药品名称,制药公司,批号,规格,商品编码,单价,数量,金额信息。
    6.注意区分收货地址与门店地址这两个字段的不同。收货地址可能为空，而门店地址通常不为空。门店地址通常在最后出现。
    </rule>

    输出格式要求:
    <format>
    1.如果没有识别到具体内容的字段，请用空字符串代替，不要生成识别以外的内容，不要编造其他字段,不要复用输出示例里面的内同。
    2.严格以json的格式直接输出内容,识别的信息以键值对填充。
    3.药品信息作为一个数组，每个药品作为数组中的一个对象，每个药品对象的键值对包括药品名称,制药公司,批号,规格,商品编码,单价,数量和金额。其中药品名称只有从药品信息中提取的药品名称，没有其他信息。
    4.请直接输出门店地址字段,不要添加任何前缀或标记，比如"地址:",如果有，请删掉。
    5.输出需要放在<answer></answer> XML标签内。
    </format>

    输出示例:
    <example>
    {   
        "门店名称": "西安怡康230523-康儿倍渭滨路店",
        "门店地址": "西安市新城区新科路8号",
        "日期": "2018-06-23 13:29:52",
        "流水号": "16349256",
        "应收金额": "72.8",
        "实付金额": "72.34",
        "优惠金额": "0.46",
        "会员卡号": "2012021147082",
        "药品信息": [
            {
                "药品名称": "多潘立酮片(吗丁啉)",
                "制药公司": "西安杨森制药有限公司",
                "批号": "180305120",
                "规格": "10MG*42粒",
                "商品编码":"306232",
                "单价": "33.12",
                "数量": "1",
                "金额": "33.12"
            },
            {
                "药品名称": "舒筋活血片",
                "制药公司": "台州南峰药业有限公司",
                "批号": "180409",
                "规格": "100ML",
                "商品编码":"50653",
                "单价": "1.57",
                "数量": "1",
                "金额": "1.57"
            }
        ]
    }
    </example>

    <input>
    %s
    </input>
    \n\nAssistant:
    """%ocr_result

    print (prompt)
    return prompt
    
def call_claude(prompt, modelId, bedrock):
    contentType = 'application/json'
    body = json.dumps({ "prompt":prompt,
                   "max_tokens_to_sample": 5000,
                   "temperature": 0.1,
                   "top_k": 250,
                   "top_p": 1,
                   "stop_sequences": []
                  }) 
    response = bedrock.invoke_model(body=body, modelId=modelId, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    return response_body.get("completion")

def get_completion(response):
    completion = response.strip(" <answer>")
    completion = completion.strip("</answer>")
    return completion
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

