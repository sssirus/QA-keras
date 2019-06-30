#-*- coding: utf-8 -*-
from predict.predicate import model
from predict.externalInterfaceNER import externalModel
from preprocess.data_preprocessor import tokenize
import numpy as np


def decode_predictions_from_candidate( preds, candidate_list):
    top_indices = preds.argmax()
    print("top_indices")
    print(top_indices)
    tag = candidate_list[top_indices]
    num = preds[top_indices]
    return tag, num
def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c
def emsemble(question, candidate):
    tag1, y1=model.predict_from_predicate(question,candidate)
    candidates = tokenize(candidate.strip())
    #y2 = externalModel.calculate_batch(question,candidates)
    #y=list_add(y1,y2)
    y=np.asarray(y1)
    tag, num = decode_predictions_from_candidate(y, candidates)

    return tag
inpute_question = "是 什么 年代 的？"
candidate = "分类 登录 类目 时代 简介 所在"
tag=emsemble(inpute_question,candidate)
print(tag.decode('string_escape'))