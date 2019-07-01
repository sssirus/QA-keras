# coding: utf-8
import re
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
    fn = io.open(os.path.join(path, "newdict.txt"), 'w',
                 encoding='UTF-8')
    f = io.open(os.path.join(path, dict_file), 'r',
                encoding='UTF-8')
    for line in f:

        try:
            values = line.split(' ')
            word = values[0]
            frequency = values[1]
            wordpropoty=values[2]

            #print(word)
        except:
            print("==============error in dict.txt:===============")
            print(line)
            continue
        dict[word] = 1
        fn.write(line)
    f.close()

    fe = io.open(os.path.join(path, entity_file), 'r',
                encoding='UTF-8')

    i=0
    for line in fe:
        #print(line)
        try:

            values = line.split(' ')
            word = values[0]
            frequency = values[1]
            wordpropoty=values[2]
            #print(line)
        except:
            print("==============error in entity.txt:===============")
            print(line)
            continue
        if word not in dict :

            i=i+1
            fn.write(line)
            #print(word)
    fe.close()


    fn.close()
    print("总共新增词数目：")
    print(i)
    return
def generateNewDictFromEntityURL(path,dict_file,entity_file):
    # 将原生分词词典中没有的实体找到,和原词典一起生成新词典
    dict = {}
    fn = io.open(os.path.join(path, "newdict.txt"), 'w',
                 encoding='UTF-8')
    f = io.open(os.path.join(path, dict_file), 'r',
                encoding='UTF-8')
    for line in f:

        try:
            values = line.split(' ')
            word = values[0]
            frequency = values[1]
            wordpropoty = values[2]

            # print(word)
        except:
            print("==============error in dict.txt:===============")
            print(line)
            continue
        dict[word] = 1
        fn.write(line)
    f.close()

    fe = io.open(os.path.join(path, entity_file), 'r',
                 encoding='UTF-8')

    i = 0
    for line in fe:
        # print(line)
        try:

            values = line.split(' ')
            word = values[0]
            url = values[1]
            #wordpropoty = values[2]
            # print(line)
        except:
            print("==============error in entity_row.txt:===============")
            print(line)
            continue
        if word not in dict:
            i = i + 1

            # print(value+" "+line)
            newstr = word + " " + "20000" + " " + "n" + "\n"
            fn.write(newstr)
            # print(word)
    fe.close()

    fn.close()
    print("总共新增词数目：")
    print(i)
    return
def generateNewDictFromEntity(path,dict_file,entity_file):
    # 将原生分词词典中没有的实体找到,和原词典一起生成新词典
    dict = {}
    fu = io.open(os.path.join(path, "entities_url.txt"), 'w',
                 encoding='UTF-8')
    fn = io.open(os.path.join(path, "newdict.txt"), 'w',
                 encoding='UTF-8')
    f = io.open(os.path.join(path, dict_file), 'r',
                encoding='UTF-8')
    for line in f:

        try:
            values = line.split(' ')
            word = values[0]
            frequency = values[1]
            wordpropoty = values[2]

            # print(word)
        except:
            print("==============error in dict.txt:===============")
            print(line)
            continue
        dict[word] = 1
        fn.write(line)
    f.close()

    fe = io.open(os.path.join(path, entity_file), 'r',
                 encoding='UTF-8')

    i = 0
    for line in fe:
        # print(line)
        try:
            rm = "\n"
            value = line.rstrip(rm)
            value = value.replace(" ", "");

            word = RemoveSpacesAndPunctuation(value)
            #url = values[1]
            #wordpropoty = values[2]
            # print(line)
        except:
            print("==============error in all_entities.txt:===============")
            print(line)
            continue
        if word not in dict and word.strip()!="":
            i = i + 1
            newstr_url=word+" "+"http://zhishi.me/baidubaike/resource/"+line
            fu.write(newstr_url)
            # print(value+" "+line)
            newstr = word + " " + "20000" + " " + "nz" + "\n"
            fn.write(newstr)
            # print(word)
    fe.close()
    fu.close()
    fn.close()
    print("总共新增词数目：")
    print(i)
    return
def RemoveSpacesAndPunctuation(str):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    str = str.strip()
    str = re.sub(r, '', str)
    return str
#generateEntity(r"/data/ylx","zhishime_all_entities.txt")
#generateEntityAndURL(r"/data/ylx","zhishime_all_entities.txt")
#findAddictionalEntity(r"/data/ylx/ylx/data/","dict.txt","entities.txt")
#generateNewDictFromEntityURL(r"/data/ylx/ylx/data/","dict.txt","entities_row.txt")
generateNewDictFromEntity(r"/data/ylx/ylx/data/","dict.txt","all_entities.txt")