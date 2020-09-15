import jieba.analyse
import numpy as np
import sys


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


# 写入文件
def write_consequence_to_file(similarity: float, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(round(similarity, 2)))


def main():
    source_path = sys.argv[1]
    copy_path = sys.argv[2]
    ans_path = sys.argv[3]
    similarity = get_similarity(source_path, copy_path)
    write_consequence_to_file(similarity, ans_path)


# 计算主过程
def get_similarity(source_path: str, copy_path: str) -> float:
    source_content = read_file(source_path)
    copy_content = read_file(copy_path)
    source_tfidf_dict = get_tfidf_dict(source_content)
    copy_tfidf_dict = get_tfidf_dict(copy_content)
    [source_tfidf_list, copy_tfidf_list] = get_tfidf_list(source_tfidf_dict, copy_tfidf_dict)
    return calculate_similarity(source_tfidf_list, copy_tfidf_list)


# 读取文件返回两个文件地址
def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    main()
