# -*- coding: utf-8 -*-
import os
import sys
import io
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
    params = "台北市立重庆国民中学是什么学校？"
    question_cut = jieba.cut(params)
    question_cut_list = list(question_cut)
    print (' '.join(question_cut_list))

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
class NER:
    dict=None
    #entity_files = "entities.txt"
    entity_url_file="entities_url.txt"
    path = "/data/ylx/ylx/data/"
    def __init__(self):
        self.dict={}
        f = io.open(os.path.join(self.path, self.entity_url_file), 'r',
                    encoding='UTF-8')
        for line in f:
            try:
                values = line.split(' ')
                word = values[0]
                url = values[1]
                rm = "\n"
                url = url.rstrip(rm)

            except:
                print(line)
                continue
            self.dict[word] = url
        f.close()
        jieba.set_dictionary(os.path.join(self.path, "newdict.txt"))
        print("词典加载完成")
        #jieba.load_userdict(os.path.join(self.path, self.entity_files))  # file_name 为文件类对象或自定义词典的路径
        self.dictBasedNER("小红帽特工队的续作是？")

    def ltp(self,str2):
        LTP_DATA_DIR = '/data/ylx/ltp_data_v3.4.0/'  # ltp模型目录的路径

        # 词性标注

        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        postagger = Postagger()  # 初始化实例
        postagger.load(pos_model_path)  # 加载模型

        print(str2)
        words=[x.decode('string_escape') for x in str2]
        # words = ['元芳', '你', '怎么', '看']  # 分词结果
        postags = postagger.postag(words)  # 词性标注

        print('\t'.join(postags))
        # ner
        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        recognizer = NamedEntityRecognizer()  # 初始化实例
        recognizer.load(ner_model_path)  # 加载模型

        netags = recognizer.recognize(words, postags)  # 命名实体识别
        netags_list = list(netags)  # 命名实体识别)  # 命名实体识别
        print('\t'.join(netags))
        res = ''
        for index in range(len(netags_list)):
            if netags_list[index].startswith('S'):
                res = str2[index]
                break;
            elif netags_list[index].startswith('B'):
                res = str2[index]
            elif netags_list[index].startswith('I'):
                res += str2[index]
            elif netags_list[index].startswith('E'):
                res += str2[index]

        if res == '':
            res = str2[0]

        print(res.decode('string_escape'))
        postagger.release()  # 释放模型
        recognizer.release()
        return res
    def dictBasedNER(self,question):
        str = " "
        res = []
        res_words=[]
        url=[]
        isfound=False
        question_cut = jieba.cut(question)
        print("分词结果")

        question_cut_list = list(question_cut)
        quesionToken = str.join(question_cut_list)
        print(quesionToken)
        #temp="小红帽特工队"
        #print(temp)
        #print(temp in self.dict.keys())
        #print(self.dict[temp].decode('utf-8'))

        for word in question_cut_list:
            if word in self.dict.keys():
                print("实体：")
                print(word)
                #print("url：")
                #print(self.dict[word])
                #print("===================")
                res_words .append(word)
                url.append(self.dict[word])

                isfound=True



        if(isfound==False):

            res_words_item = self.ltp(question_cut_list)
            res_words.append(res_words_item)
            #res.append(str.join(res_item))
            url.append("None")

        for x in res_words:

            removed = question_cut_list[:]
            removed.remove(x)
            res.append(str.join(removed))
        print("找到的实体：")
        for x in res_words:
            print(x)
        print("剩余部分:")
        for x in res:
            print(x.decode('string_escape'))
        return res_words,url,res

ner = NER()
#ltp()