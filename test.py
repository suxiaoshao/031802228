# 导入所需的模块
import jieba.analyse

f1 = "C:\\Users\\ss\\Desktop\\文本相似度\\sim_0.8\\orig.txt"
f2 = "C:\\Users\\ss\\Desktop\\文本相似度\\sim_0.8\\orig_0.8_dis_1.txt"

content1 = open(f1, encoding='UTF-8').read()
content2 = open(f2, encoding='UTF-8').read()
contents = content1 and content2
one_words = []
one_tfidf = {}
two_words = []
two_tfidf = {}
for word, tfidf in jieba.analyse.extract_tags(content1, topK=0, withWeight=True):
    one_words.append(word)
    one_tfidf[word] = tfidf
print(one_words)
print(one_tfidf)
for word, tfidf in jieba.analyse.extract_tags(content2, topK=0, withWeight=True):
    two_words.append(word)
    two_tfidf[word] = tfidf
