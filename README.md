# QA-kera

该项目实现了中文问答系统，该模型大体借鉴了《*An Attention-Based Word-Level Interaction Model:Relation Detection for Knowledge Base Question Answering*》提出的模型。采用Tensorflow 后台的 Keras 实现。

- main.py 是程序的入口，用于训练模型
- loadModel：模型存放在这个文件
- globalvar.py 是实现全局变量的模块，模型中的参数作为全局变量维护
- attention.py 是基于论文实现的Attention模块，根据论文中的公式实现了Attention类
- data_preprocessor.py 是处理数据的模块，将输入数据进行分割等预处理，以便于进行Embedding
- predicate：采用训练好的模型进行预测
- cnn，lstm_cnn, attention_lstm_cnn：几种不同的模型进行对比

用法如下：

1. 训练模型：main.py
2. 用模型预测：predict_anwser.py
3. 启动flask服务：flask_keras.py