#-*- coding: utf-8 -*-
from predict.predicate import model
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