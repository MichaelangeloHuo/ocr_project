# coding:utf-8
import xlrd
import collections
import time
import jiebaocr
from gensim import corpora, models, similarities


#jiebaocr为自己修改过的
def match(query, delta):
    start = time.time()
    workbook = xlrd.open_workbook('./testData.xlsx')
    booksheet = workbook.sheet_by_index(0)
    #print(type(booksheet))
    rawTexts = collections.OrderedDict()
    for i in range(1, 74):
        id = booksheet.row_values(i)[0]
        text = booksheet.row_values(i)[1]
        rawTexts[id] = text
        # print(booksheet.row_values(i))
        jiebaocr.load_userdict("./dict.txt")
    # print(texts)
    texts = []
    codeList = []
    for key, value in rawTexts.items():
        texts.append(list("".join(jiebaocr.cut(value))))
        #print("/".join(jiebaocr.cut(value)))
        codeList.append(key)

    dictionary = corpora.Dictionary(texts)
    #print('=======\n', dictionary)
    #print('=======\n', dictionary.token2id)
    featureNum = len(dictionary.token2id.keys())  # 提取词典特征数
    dictionary.save("./dict_match")

    corpus = [dictionary.doc2bow(text) for text in texts]
    #print('\n==================================\n', corpus)
    #print(corpus[0])
    tfidf = models.TfidfModel(corpus)

    # 9.计算相似性
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featureNum)
    end = time.time()
    print("loading BOW model cost {:.3f} seconds" .format(end - start))

    start = time.time()
    # 7.加载句子并整理其格式
    #query = "4。下图中正方形的周长是32厘米,求阴影平行四边形的面积。"
    dataQ = "".join(jiebaocr.cut(query))
    print("/".join(jiebaocr.cut(query)))
    dataQuery = ""
    for item in dataQ:
        dataQuery += item + " "
    new_doc = dataQuery
    #print(new_doc)
    # 8.将对比句子转换为稀疏向量
    new_vec = dictionary.doc2bow(new_doc.split())
    #print(type(index))
    # print(index.get_similarities())
    print(tfidf[new_vec])
    # 9.计算句子的相似度
    simText = index[tfidf[new_vec]]
    # sort_sim = sorted(simText)
    simAll = [i for i in zip(codeList, simText)]
    # 将相似度存入元组，并降序排列
    simAllSorted = sorted(simAll, key=lambda x: (x[1]), reverse=True)
    # 输出top-5
    for target in simAllSorted[:10]:
        #if sim[i] > 0.3:
        print("查询与第%s题相似度为%f" % (target[0], target[1]))
        #pass

    #判断是否有多个连续的值
    topOne = simAllSorted[0][1]#存储最大的那个值
    topN = []
    for i in simAllSorted:
        #x, y = simAllSort[i], simAllSort[i + 1]
        #d = abs(x[1] - y[1])#相似度值相减
        dOne = abs(i[1] - topOne)
        if dOne <= delta:
            topN.append(i)
        else:
            break
    #print('---------------------------')
    if topN[0][1] <= 0.8:
        #图片切割，并返回0或者1，1表示有图像
        pass

    for target in topN:
        #if sim[i] > 0.3:
        print("查询与第%s题相似度为%f" % (target[0], target[1]))

    end = time.time()
    print("matching cost {:.3f} seconds".format(end - start))