# -*- coding: utf-8 -*-
import os
import sys
from imp import reload
import jieba
reload(sys)
sys.setdefaultencoding('utf8')

from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Segmentor
def ltp():
    LTP_DATA_DIR = '/data/ylx/ltp_data_v3.4.0/'  # ltp模型目录的路径
    #分词
    str = " "
    params = "地球的半长轴有多长？"

    #question_cut = jieba.cut(params)
    #question_cut_list = list(question_cut)
    #quesionToken = str.join(question_cut)
    #print(quesionToken)

    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`


    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型


    question_cut = segmentor.segment(params)  # 分词
    question_cut_list = list(question_cut)
    print ('\t'.join(question_cut))
    str2 =[x.decode('string_escape') for x in question_cut_list]

    #print(question_list)
    #print(question_list)
    #词性标注


    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    #words = ['元芳', '你', '怎么', '看']  # 分词结果
    postags = postagger.postag(str2)  # 词性标注

    print ('\t'.join(postags))
    #ner
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
    recognizer = NamedEntityRecognizer() # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    netags = recognizer.recognize(str2, postags)  # 命名实体识别
    netags_list = list(netags)  # 命名实体识别)  # 命名实体识别
    print ('\t'.join(netags))
    res=''
    for i in netags_list:
        index = netags_list.index(i)
        if i.startswith('S'):
            res=question_cut_list[index]
            break;
        elif i.startswith('B'):
            res = question_cut_list[index]
        elif i.startswith('I'):
            res += question_cut_list[index]
        elif i.startswith('E'):
            res += question_cut_list[index]
            break




    print(res.decode('string_escape'))
    postagger.release()  # 释放模型
    recognizer.release()
    return
