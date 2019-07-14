#-*- coding: utf-8 -*-
from predict.model_ensemble import emsemble
from predict.predicate import model
from ner.nerModel import ner
from flask import Flask,jsonify,request
import jieba

app = Flask(__name__)
@app.route('/predict', methods=["POST"])
def predict():
    params1 = request.form.get('question')
    params2 = request.form.get('cadidate')
    # 若发现参数，则返回预测值
    if (params1 != None and params2!= None):

        str = " "
        print("候选谓词：")
        print(params2)
        #entitystr, urlstr, quesionToken = ner.dictBasedNER(params)
        #question_list = jieba.cut(params)

        #quesionToken = str.join(question_list)
        #print(quesionToken)
        tags,score = emsemble(params1,params2)
        print(tags)
        my_dict = { "predicate": tags,"score":float(score[0])}
    # 返回响应
    return jsonify(my_dict)
@app.route('/entity', methods=["POST"])
def entity():
    params = request.form.get('question')

    # 若发现参数，则返回预测值
    if (params != None):

        str = " "
        print(params)
        entitystr, urlstr, remain = ner.dictBasedNER(params)
        #question_list = jieba.cut(params)

        #quesionToken = str.join(question_list)
        #print(quesionToken)
        #tags = model.predicated_quick(quesionToken)
        #for tag in tags:
            #print(tag)
        my_dict = {"entity": entitystr, "url":urlstr,"remain":remain}
    # 返回响应
    return jsonify(my_dict)

def predict_old():
    params = request.form.get('question')

    # 若发现参数，则返回预测值
    if (params != None):
        str = " "
        print(params)
        question_list = jieba.cut(params)

        quesionToken = str.join(question_list)
        print(quesionToken)
        tags = model.predicated_quick(quesionToken)
        for tag in tags:
            print(tag)
        my_dict = {"entity": "", "predicate": tags}
    # 返回响应
    return jsonify(my_dict)
# 當啟動 server 時先去預先 load model 每次 request 都要重新 load 造成效率低下且資源浪費
if __name__ == '__main__':
     app.run(debug=False, host='0.0.0.0', port=6008)

