# 论文相似度计算系统

## 信息

[github 项目地址](https://github.com/suxiaoshao/031802228)

## 计算模块接口的设计与实现过程

看到这个题目就使用了搜索引擎搜了一下，大部分是使用 jieba + gensim 的方式，
研究了好几天没有搞懂，后来看了
[基于 TF-IDF、余弦相似度算法实现文本相似度算法的 Python 应用](https://www.pythonf.cn/read/39910)
才知道可以不用 gensim 库。

具体步骤如下

![uml](https://img2020.cnblogs.com/blog/2145446/202009/2145446-20200916135840039-1448668603.png)

搞懂了之后就比较简单了，打出代码

```python
from typing import List, Dict

import jieba.analyse
import numpy as np
import sys


# 计算主过程
def get_similarity(source_path: str, copy_path: str) -> float:
    source_content = read_file(source_path)
    copy_content = read_file(copy_path)
    source_tfidf_dict = get_tfidf_dict(source_content)
    copy_tfidf_dict = get_tfidf_dict(copy_content)
    [source_tfidf_list, copy_tfidf_list] = get_tfidf_list(
        source_tfidf_dict, copy_tfidf_dict)
    return calculate_similarity(source_tfidf_list, copy_tfidf_list)


# 读取文件返回两个文件地址
def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


# 获取一个文本的 tfidf dict
def get_tfidf_dict(content: str) -> Dict[str, float]:
    tfidf_dict = {}
    for word, tfidf in jieba.analyse.extract_tags(content, topK=0, withWeight=True):
        tfidf_dict[word] = tfidf
    return tfidf_dict


# 获取两个 tfidf dict 的 list
def get_tfidf_list(source_tfidf_dict: Dict[str, float], copy_tfidf_dict: Dict[str, float]) -> List[List[float]]:
    source_tfidf_list = []
    copy_tfidf_list = []
    for item in source_tfidf_dict:
        source_tfidf_list.append(source_tfidf_dict[item])
        copy_tfidf_list.append(
            copy_tfidf_dict[item] if item in copy_tfidf_dict else 0)
    for item in copy_tfidf_dict:
        if item not in source_tfidf_dict:
            source_tfidf_list.append(0)
            copy_tfidf_list.append(copy_tfidf_dict[item])
    return [source_tfidf_list, copy_tfidf_list]


# 计算结果
def calculate_similarity(source_tfidf_list: List[float], copy_tfidf_list: List[float]) -> float:
    source_tfidf_array = np.array(source_tfidf_list)
    copy_tfidf_array = np.array(copy_tfidf_list)
    return np.dot(source_tfidf_array, copy_tfidf_array, out=None) / (
            np.linalg.norm(source_tfidf_array) * np.linalg.norm(copy_tfidf_array))


# 写入文件
def write_consequence_to_file(similarity: float, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(round(similarity, 2)))


def main():
    [source_path, copy_path, ans_path] = sys.argv[1:]
    similarity = get_similarity(source_path, copy_path)
    write_consequence_to_file(similarity, ans_path)


if __name__ == '__main__':
    main()

```

在 main 函数里使用 sys 获取三个文件 ,然后进入 get_similarity 函数，通过 read_file 函数获取前两个文件的内容。

然后进入 get_tfidf_dict 获取他们的词频,再在 get_tfidf_list 获取
词频的有序列表,最后使用 calculate_similarity 获取余弦相似度。

最后用 write_consequence_to_file 把答案写入文件

## 计算模块接口部分的性能改进

用 pycharm 自带的 profile （pycharm 永远滴神）分析了性能，生成了以下图片。

![性能示例图](https://img2020.cnblogs.com/blog/2145446/202009/2145446-20200916141957773-1289666845.png)

可以看出这个性能还不错,2000ms 左右就能完成,main 函数也没有占用太多的性能,主要是 python 导包的性能确实,main 函数内部主要是 get_tfidf_dict 耗费了太多资源,我就想到了使用 pandas 这个包来计算词频。

```python
# 获取一个文本的 tfidf dict
def get_tfidf_dict(content: str) -> Dict[str, float]:
    # tfidf_dict = {}
    # for word, tfidf in jieba.analyse.extract_tags(content, topK=0, withWeight=True):
    #     tfidf_dict[word] = tfidf
    # return tfidf_dict
    data = pd.Series(jieba.cut(content))
    return dict(data.value_counts())
```

把 get_tfidf_dict 改成这样,再使用 profile 发现性能没有差别，所以这个应该是 jieba 分词的时间消耗,难以优化，考虑这样性能就可以了,我就没有继续优化。

![优化后](https://img2020.cnblogs.com/blog/2145446/202009/2145446-20200916154022199-1041502967.png)
