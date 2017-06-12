'''
如果用词向量计算句子相似度，请下载3个文件：Word60.model，Word60.model.syn0.npy，Word60.model.syn1neg.npy
下载地址：http://pan.baidu.com/s/1kURNutT 密码：1tq1

robot.answer(inputtxt,'simple_POS')   simType参数有如下模式：
simple：简单的对比相同词汇数量，得到句子相似度
simple_POS：简单的对比相同词汇数量,并对词性乘以不同的权重，得到句子相似度
vec：用词向量计算相似度,并对词性乘以不同的权重，得到句子相似度
all：调试模式，把以上几种模式的结果都显示出来，方便对比和调试

inputtxt可输入的特殊文本命令：
-zsk 显示当前知识库
-s -1 查看上一个问句的结果和中间参数
-q -1 重复提问，把当一个问句当做输入
-reload 重新载入QA知识库
'''

import logging
import time

import jieba
import jieba.posseg as pseg


lg = logging.getLogger('ZL')
lg.addHandler(logging.StreamHandler())
jieba.default_logger.setLevel(logging.ERROR)


def pp(ppath):
    import os
    return os.path.join(os.getcwd(), os.path.dirname(__file__), ppath)


def addlog(txt, fileName='log.txt'):
    with open(pp(fileName), 'a', encoding='utf-8') as f:
        f.write(txt)


class zhishiku(object):
    def __init__(self, q):  # a是答案（必须是1给）, q是问题（1个或多个）
        self.q = [q]
        self.a = ""
        self.sim = 0
        self.q_vec = []
        self.q_word = []

    def __str__(self):
        return 'q=' + str(self.q) + '\na=' + str(self.a) + '\nq_word=' + str(self.q_word) + '\nq_vec=' + str(self.q_vec)
        # return 'a=' + str(self.a) + '\nq=' + str(self.q)


class FAQrobot(object):
    def __init__(self, zhishitxt='FAQ_减肥.txt', lastTxtLen=10, usedVec=False):
        # usedVec 如果是True 在初始化时会解析词向量，加快计算句子相似度的速度
        self.lastTxt = []  # 记录之前输入的问句，方便调试
        self.lastTxtLen = lastTxtLen  # lastTxt数组的长度上限
        self.zhishitxt = zhishitxt
        self.posWeight = {
            "Ag": 1,  # 形语素
            "a": 0.5,  # 形容词
            "ad": 0.5,  # 副形词
            "an": 1,  # 名形词
            "b": 1,  # 区别词
            "c": 0.2,  # 连词
            "dg": 0.5,  # 副语素
            "d": 0.5,  # 副词
            "e": 0.5,  # 叹词
            "f": 0.5,  # 方位词
            "g": 0.5,  # 语素
            "h": 0.5,  # 前接成分
            "i": 0.5,  # 成语
            "j": 0.5,  # 简称略语
            "k": 0.5,  # 后接成分
            "l": 0.5,  # 习用语
            "m": 0.5,  # 数词
            "Ng": 1,  # 名语素
            "n": 1,  # 名词
            "nr": 1,  # 人名
            "ns": 1,  # 地名
            "nt": 1,  # 机构团体
            "nz": 1,  # 其他专名
            "o": 0.5,  # 拟声词
            "p": 0.3,  # 介词
            "q": 0.5,  # 量词
            "r": 0.2,  # 代词
            "s": 1,  # 处所词
            "tg": 0.5,  # 时语素
            "t": 0.5,  # 时间词
            "u": 0.5,  # 助词
            "vg": 0.5,  # 动语素
            "v": 1,  # 动词
            "vd": 1,  # 副动词
            "vn": 1,  # 名动词
            "w": 0.01,  # 标点符号
            "x": 0.5,  # 非语素字
            "y": 0.5,  # 语气词
            "z": 0.5,  # 状态词
            "un": 0.3  # 未知词
        }
        self.usedVec = usedVec
        self.reload()

    def reload(self):
        print('问答知识库开始载入')
        self.zhishiku = []
        with open(pp(self.zhishitxt), encoding='utf-8') as f:
            txt = f.readlines()
            abovetxt = 0  # 上一行的种类： 0空白/注释  1答案   2问题
            for t in txt:  # 读取FAQ文本文件
                t = t.strip()
                if t[:1] == '#' or t == '':
                    abovetxt = 0
                elif abovetxt != 2:
                    if t[:4] == '【问题】':  # 输入第一个问题
                        self.zhishiku.append(zhishiku(t[4:]))
                        abovetxt = 2
                    else:  # 输入答案文本（非第一行的）
                        self.zhishiku[-1].a += '\n' + t
                        abovetxt = 1
                else:
                    if t[:4] == '【问题】':  # 输入问题（非第一行的）
                        self.zhishiku[-1].q.append(t[4:])
                        abovetxt = 2
                    else:  # 输入答案文本
                        self.zhishiku[-1].a += t
                        abovetxt = 1

        for t in self.zhishiku:
            for question in t.q:
                t.q_word.append(set(jieba.cut(question)))
        if self.usedVec:
            print('正在载入词向量')
            from gensim.models import Word2Vec
            # 载入60维的词向量(Word60.model，Word60.model.syn0.npy，Word60.model.syn1neg.npy）
            self.vecModel = Word2Vec.load(pp('Word60.model'))
            for t in self.zhishiku:
                t.q_vec = []
                for question in t.q_word:
                    t.q_vec.append(
                        {t for t in question if t in self.vecModel.index2word})
        print('问答知识库载入完毕')

    def printKu(self):
        for t in self.zhishiku:
            print(t, '\n')

    def intolastTxt(self, intxt):
        self.lastTxt.append(intxt)
        if len(self.lastTxt) > self.lastTxtLen:
            self.lastTxt.pop(0)

    # 找出知识库里的和输入句子相似度最高的句子
    # simType=simple, simple_POS, vec
    def maxSimTxt(self, intxt, simCondision=0.1, simType='simple'):
        self.intolastTxt(intxt)
        for t in self.zhishiku:
            simList = []
            questionMaxSim = 0
            if simType == 'vec':
                if not self.usedVec:
                    self.usedVec = True
                    self.reload()
                for question in t.q_vec:
                    simValue = self.juziSim_vec(intxt, question)
                    if questionMaxSim < simValue:
                        questionMaxSim = simValue
            elif simType == 'simple':
                for question in t.q_word:
                    simValue = self.juziSim_simple(intxt, question)
                    if questionMaxSim < simValue:
                        questionMaxSim = simValue
            elif simType == 'simple_POS':
                for question in t.q_word:
                    simValue = self.juziSim_simple_POS(intxt, question)
                    if questionMaxSim < simValue:
                        questionMaxSim = simValue
            else:
                return 'error:  maxSimTxt的simType类型不存在:' + str(simType)
            t.sim = questionMaxSim
        maxSim = max(self.zhishiku, key=lambda x: x.sim)
        lg.info('maxSim=' + format(maxSim.sim, '.0%'))
        if maxSim.sim < simCondision:
            return '抱歉，我没有理解您的意思。请您询问有关减肥的话题。'

        return maxSim.a

    def juziSim_simple(self, intxt, questionWordset):
        """
        简单的对比相同词汇数量，得到句子相似度
        参数：juziIn, 输入的句子，juziLi句子库里的句子
        """
        intxtSet = set(jieba.cut(intxt))
        if not intxtSet:
            return 0
        simWeight = 0
        for t in intxtSet:
            if t in questionWordset:
                simWeight += 1
        return simWeight / len(intxtSet)

    # simple_POS: 简单的对比相同词汇数量,并对词性乘以不同的权重，得到句子相似度
    # juziIn输入的句子，juziLi句子库里的句子
    def juziSim_simple_POS(self, intxt, questionWordset, posWeight=None):
        if posWeight is None:
            posWeight = self.posWeight
        intxtSet = set(pseg.cut(intxt))
        if not intxtSet:
            return 0
        simWeight = 0
        totalWeight = 0
        for word, pos in intxtSet:
            wordPosWeight = posWeight.get(pos, 1)
            totalWeight += wordPosWeight
            if word in questionWordset:
                simWeight += wordPosWeight
        if totalWeight == 0:
            return 0
        return simWeight / totalWeight

    # vec: 用词向量计算相似度,并对词性乘以不同的权重，得到句子相似度
    def juziSim_vec(self, intxt, questionWordset, posWeight=None):  # juziIn输入的句子，juziLi句子库里的句子
        if posWeight is None:
            posWeight = self.posWeight
        intxtSet = set(list(pseg.cut(intxt)))
        if not intxtSet:
            return 0
        simWeight = 0
        totalWeight = 0
        for word, pos in intxtSet:
            if word in self.vecModel.index2word:
                wordPosWeight = posWeight.get(pos, 1)
                totalWeight += wordPosWeight

                wordMaxWeight = 0
                for t in questionWordset:
                    tmp = self.vecModel.similarity(word, t)
                    if wordMaxWeight < tmp:
                        wordMaxWeight = tmp
                simWeight += wordPosWeight * wordMaxWeight
        if totalWeight == 0:
            return 0
        return simWeight / totalWeight

    def answer(self, intxt, simType='simple'):  # simType=simple, simple_POS, vec, all
        if intxt == '-zsk':  # '-zsk'  显示当前所有知识库
            self.printKu()
            return ''
        elif intxt == '-reload':  # -reload 重新载入QA知识库
            self.reload()
            return 'reload完毕'
        elif intxt[:3] == '-s ':  # -s -1 查看上一个问句的结果和中间参数
            return 'pass,   error'
        else:
            if intxt[:3] == '-q ':  # -q -1 重复提问，把当一个问句当做输入
                tmp = intxt.split(' ')
                try:
                    intxt = self.lastTxt[int(tmp[-1])]
                except (IndexError, ValueError) as e:
                    print(e)
                    addlog(time.strftime('%F %R') + '\n输入：' +
                           intxt + '\n' + str(e) + '\n\n')
                    return ''
            if not intxt:
                return ''
            # 以上是处理特殊命令，以下是计算回复内容
            if simType == 'all':  # 用于测试不同类型方法的准确度，返回空文本
                addlog(time.strftime('%F %R') + '\n输入：' + intxt + '\n')
                # simple
                outtxt = 'simple:\t' + self.maxSimTxt(intxt, simType='simple')
                print(outtxt)
                addlog(outtxt + '\n')
                # simple_POS
                outtxt = 'simple_POS:\t' + \
                    self.maxSimTxt(intxt, simType='simple_POS')
                print(outtxt)
                addlog(outtxt + '\n')
                # vec
                outtxt = 'vec:\t' + self.maxSimTxt(intxt, simType='vec')
                print(outtxt)
                addlog(outtxt + '\n')
                return ''
            else:
                outtxt = self.maxSimTxt(intxt, simType=simType)
            # 输出回复内容，并计入日志
            addlog(time.strftime('%F %R') + '\n输入：' +
                   intxt + '\n回复：' + outtxt + '\n\n')
            return outtxt


if __name__ == '__main__':
    # lg.setLevel(logging.DEBUG)
    robot = FAQrobot('FAQ_减肥.txt', usedVec=False)
    while True:
        # simType=simple, simple_POS, vec, all
        print('回复：' + robot.answer(input('输入：'), 'simple_POS') + '\n')
