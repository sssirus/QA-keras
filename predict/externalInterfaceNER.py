#-*- coding: utf-8 -*-

from aip import AipNlp

class baidunlp:
    """ 你的 APPID AK SK """
    APP_ID = '16609672'
    API_KEY = 'UwXfuFVFvKWTWxyKS0PRlS68'
    SECRET_KEY = 'N1u3d2czK04Ka1LqYNTKMSVLLHB5RcTT'
    client=None
    def __init__(self):
        self.client=self.load_data()
        self.calculate_one("浙富股份","万事通自考网")
    def load_data(self):
        client = AipNlp(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        return  client
    def calculate_one(self,str1,str2):
        """ 调用短文本相似度 """
        res=self.client.simnet(str1, str2)
        print(res.get("score"))
        return res.get("score")
    def calculate_batch(self,question,candidate):
        return [self.calculate_one(question,x) for x in candidate ]

externalModel=baidunlp()
