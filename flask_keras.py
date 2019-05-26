#-*- coding: utf-8 -*-
from predicate import model
from flask import Flask,jsonify,request
import jieba

app = Flask(__name__)
@app.route('/predict', methods=["POST"])
def predict():
    params = request.form.get('question')

    # 若发现参数，则返回预测值
    if (params != None):
        str = " ";
        print(params)
        question_list = jieba.cut(params)

        quesionToken = str.join(question_list)
        print(quesionToken)
        tag = model.predicated_quick(quesionToken)
        print(tag)
        my_dict = {"entity": "", "predicate": tag}
    # 返回响应
    return jsonify(my_dict)


# 當啟動 server 時先去預先 load model 每次 request 都要重新 load 造成效率低下且資源浪費
if __name__ == '__main__':
     app.run(debug=False, host='0.0.0.0', port=6006)

