#-*- coding: utf-8 -*-
from predicate import model
import time

inpute_question = "创始人 是 谁"
print(inpute_question)
start =time.clock()
tag=model.predicated_quick(inpute_question)
print(tag)
end = time.clock()
print('Running time: %s Seconds'%(end-start))

inpute_question = "创始人 是 谁"
print(inpute_question)
start =time.clock()
tag=model.predicated_quick(inpute_question)
print(tag)
end = time.clock()
print('Running time: %s Seconds'%(end-start))

inpute_question = "是 什么 年代 的？"
candidate = "分类 登录 类目 时代 简介 所在"
tag=model.predict_from_predicate(inpute_question,candidate)
print(tag.decode('string_escape'))
