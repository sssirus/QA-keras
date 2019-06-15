#-*- coding: utf-8 -*-
import urllib.parse
import os
import io

def generateEntity(path,filename):
    i=0
    f = io.open(os.path.join(path, filename), 'r', encoding='UTF-8')
    fw = io.open(os.path.join(path, "entities.txt"), 'w', encoding='utf-8')
    for line in f:
        x = line.split("resource/", 1)
        value=urllib.parse.unquote(x[1])
        print(value)
        fw.write(value)
        i=i+1
    f.close()
    fw.close()
    print("总共实体数：")
    print(i)
    return
def generateEntityAndURL(path,filename):
    i=0
    f = io.open(os.path.join(path, filename), 'r', encoding='UTF-8')
    fw = io.open(os.path.join(path, "entitiesURL.txt"), 'w', encoding='utf-8')
    for line in f:
        x = line.split("resource/", 1)
        value=urllib.parse.unquote(x[1])
        rm="\n"
        value=value.rstrip(rm)
        print(value+" "+line)
        fw.write(value+" "+line)
        i=i+1
    f.close()
    fw.close()
    print("总共实体数：")
    print(i)
    return
generateEntity(r"/data/ylx","zhishime_all_entities.txt")
generateEntityAndURL(r"/data/ylx","zhishime_all_entities.txt")