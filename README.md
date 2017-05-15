# FAQrobot
一个自动回复FAQ问题的聊天机器人。目前使用了简单词汇对比、词性权重、词向量3种相似度计算模式。输入符合格式的FAQ文本文件即可立刻使用。欢迎把无法正确区分的问题和FAQ文件发送到评论区。
 
## 依赖的库
jieba 分词使用的库。
gensim  词向量使用的库，如果使用词向量vec模式，则需要载入。

## 依赖的文件
如果使用词向量vec模式，需要下载3个文件：Word60.model，Word60.model.syn0.npy，Word60.model.syn1neg.npy
下载地址：http://pan.baidu.com/s/1kURNutT 密码：1tq1



