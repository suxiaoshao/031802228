import sys

import jieba


def get_all_cut(source_cut: dict, copy_cut: dict) -> dict:
    result = {}
    for item in source_cut:
        result[item] = {
            "source_count": source_cut[item],
            "copy_count": copy_cut[item] if item in copy_cut else 0
        }
    for item in copy_cut:
        if item not in result:
            result[item] = {
                "source_count": 0,
                "copy_count": copy_cut[item]
            }
    return result


def cut_content(content: str) -> dict:
    result_list = list(jieba.cut(content))
    filter_list = ['\n', '，', '？', ' ', '、', '《', '》', '。', '“', '”', '；', '：', '’', '！']
    result_list = list(filter(lambda x: x not in filter_list, result_list))
    result_dict = {}
    for item in result_list:
        result_dict[item] = result_dict[item] + 1 if item in result_dict else 1
    return result_dict


def main():
    [source_content, copy_content] = read_file()
    ans_file = sys.argv[3]
    source_cut = cut_content(source_content)
    copy_cut = cut_content(copy_content)
    all_cut = get_all_cut(source_cut, copy_cut)
    print(all_cut)


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
