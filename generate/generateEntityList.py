# coding: utf-8
import urllib
import sys

from imp import reload
import os
import io

from numpy import unicode

reload(sys)
sys.setdefaultencoding('utf8')
def generateEntity(path,filename):
    i=0
    f = io.open(os.path.join(path, filename), 'r', encoding='UTF-8')
    fw = io.open(os.path.join(r"/data/ylx/ylx/data/", "entities.txt"), 'w', encoding='UTF-8')
    for line in f:
        x = line.split("resource/", 1)
        value = str(x[1])
        value=urllib.unquote(value)
        value = value.replace(" ", "·");
        #print(value)
        rm = "\n"
        value = value.rstrip(rm)

        # print(value+" "+line)
        newstr=value+ " " + "20000"+" "+"n"+"\n"

        fw.write(unicode(newstr, 'UTF-8'))
        i=i+1
    f.close()
    fw.close()
    print("总共实体数：")
    print(i)
    return
def generateEntityAndURL(path,filename):
    i=0
    f = io.open(os.path.join(path, filename), 'r', encoding='UTF-8')
    fw = io.open(os.path.join(r"/data/ylx/ylx/data/", "entitiesURL.txt"), 'w', encoding='utf-8')
    for line in f:
        x = line.split("resource/", 1)
        value = str(x[1])
        value=urllib.unquote(value)
        value = value.replace(" ", "·");
        rm="\n"
        value=value.rstrip(rm)
        #print(value+" "+line)
        fw.write(value+" "+line)
        i=i+1
    f.close()
    fw.close()
    print("总共实体数：")
    print(i)
    return
def findAddictionalEntity(path,dict_file,entity_file):
    # 将原生分词词典中没有的实体找到,和原词典一起生成新词典
    dict={}

    f = io.open(os.path.join(path, dict_file), 'r',
                encoding='UTF-8')
    for line in f:

        try:
            values = line.split(' ',1)
            word = values[0]
            other = values[1]

            #print(word)
        except:
            print(line)
            continue
        dict[word] = other
    f.close()

    fe = io.open(os.path.join(path, entity_file), 'r',
                encoding='UTF-8')

    i=0
    for line in fe:
        #print(line)
        try:

            values = line.split(' ',1)
            word = values[0]
            other = values[1]
            #print(line)
        except:
            #print(line)
            continue
        if word not in dict :
            dict[word] = other
            i=i+1
            #print(word)
    fe.close()
    fn = io.open(os.path.join(path, "newdict.txt"), 'w',
                 encoding='UTF-8')
    for e in dict:
        fn.write(e + " " + dict[e])
    fn.close()
    print("总共新增词数目：")
    print(i)
    return
#generateEntity(r"/data/ylx","zhishime_all_entities.txt")
#generateEntityAndURL(r"/data/ylx","zhishime_all_entities.txt")
findAddictionalEntity(r"/data/ylx/ylx/data/","dict.txt","entities.txt")