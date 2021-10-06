from flask import Flask, render_template, request, jsonify, make_response, Response
import json
import requests

from models.tools import load_config
from system import QnA_system


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    # 받아오기
    id_ = request.form['input_id']
    que_ = request.form['input_que']
    print(id_, que_)
    
    # Load config dict
    config_path = "./Base.yaml"
    cfg = load_config(config_path)
    
    # Q&A system 실행
    if (id_ != None) and (que_ != None):
        df, df1 = QnA_system(cfg, id_, que_)
        qna_dict = df.to_dict()
        review_dict = df1.to_dict()
        print('Q&A system done')
    
    # 값 넘기기
    data = jsonify({'qna_':qna_dict, 'review_':review_dict})
    return make_response(data, 201)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
