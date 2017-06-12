import logging
from os.path import join, dirname


POS_WEIGHT = {
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


def get_logger(name, logfile=None):
    """
    name: logger 的名称，建议使用模块名称
    logfile: 日志记录文件，如无则输出到标准输出
    """
    formatter = logging.Formatter(
        '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
    )

    if not logfile:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(logfile)

    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger


def similarity(a, b, method='simple', pos_weight=None, embedding=None):
    """a 和 b 是同类型的可迭代对象，比如都是词的 list"""
    if not a or not b:
        return 0

    pos_weight = pos_weight or POS_WEIGHT
    if method == 'simple':
        # 词重叠率
        return len(set(a) & set(a)) / len(set(a))

    elif method == 'simple_pos':
        # 带词性权重的词重叠率
        sim_weight = 0
        for word, pos in set(a):
            sim_weight += pos_weight.get(pos, 1) if word in b else 0

        total_weight = sum(pos_weight.get(pos, 1) for _, pos in set(a))
        return sim_weight / total_weight if total_weight > 0 else 0

    elif method == 'vec' and embedding:
        # 词向量+词性权重
        sim_weight = 0
        total_weight = 0
        for word, pos in a:
            if word not in embedding.index2word:
                continue

            cur_weight = pos_weight.get(pos, 1)

            max_word_sim = max(embedding.similarity(bword, word)
                               for bword in b)
            sim_weight += cur_weight * max_word_sim

            total_weight += cur_weight

        return sim_weight / total_weight if total_weight > 0 else 0
