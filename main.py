import sys

import jieba.analyse
import numpy as np


# 获取一个文本的 tfidf dict
def get_tfidf_dict(content: str) -> {float}:
    tfidf_dict = {}
    for word, tfidf in jieba.analyse.extract_tags(content, topK=0, withWeight=True):
        tfidf_dict[word] = tfidf
    return tfidf_dict


# 获取两个 tfidf dict 的 list
def get_tfidf_list(source_tfidf_dict: {float}, copy_tfidf_dict: {float}) -> [[float]]:
    source_tfidf_list = []
    copy_tfidf_list = []
    for item in source_tfidf_dict:
        source_tfidf_list.append(source_tfidf_dict[item])
        copy_tfidf_list.append(copy_tfidf_dict[item] if item in copy_tfidf_dict else 0)
    for item in copy_tfidf_dict:
        if item not in source_tfidf_dict:
            source_tfidf_list.append(0)
            copy_tfidf_list.append(copy_tfidf_dict[item])
    return [source_tfidf_list, copy_tfidf_list]


# 计算结果
def calculate_similarity(source_tfidf_list: [float], copy_tfidf_list: [float]) -> float:
    source_tfidf_array = np.array(source_tfidf_list)
    copy_tfidf_array = np.array(copy_tfidf_list)
    return np.dot(source_tfidf_array, copy_tfidf_array, out=None) / (
            np.linalg.norm(source_tfidf_array) * np.linalg.norm(copy_tfidf_array))


def write_consequence_to_file(similarity: float, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(similarity))


def main():
    [source_content, copy_content] = read_file()
    ans_file = sys.argv[3]
    source_tfidf_dict = get_tfidf_dict(source_content)
    copy_tfidf_dict = get_tfidf_dict(copy_content)
    [source_tfidf_list, copy_tfidf_list] = get_tfidf_list(source_tfidf_dict, copy_tfidf_dict)
    similarity = calculate_similarity(source_tfidf_list, copy_tfidf_list)
    write_consequence_to_file(similarity, ans_file)


# 读取文件返回两个文件地址
def read_file() -> [str]:
    source_file = sys.argv[1]
    copy_file = sys.argv[2]
    with open(source_file, 'r', encoding='utf-8') as f:
        source_content = f.read()
    with open(copy_file, 'r', encoding='utf-8') as f:
        copy_content = f.read()
    return [source_content, copy_content]


if __name__ == '__main__':
    main()
