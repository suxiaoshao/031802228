import os

import main


def test():
    copy_paths = './copy'
    origin_path = './origin.txt'
    for path in os.listdir(copy_paths):
        print(f'{path}:', main.get_similarity(f'{copy_paths}/{path}', origin_path))

    # 参数缺失
    res = os.system(f'python main.py {origin_path}')
    print(res)

    # 无效文本
    res = os.system(f'python main.py {origin_path} ./empty.txt ./result.txt')
    print(res)

    # 文件不存在
    res = os.system(f'python main.py ./2.txt ./empty.txt ./result.txt')
    print(res)


if __name__ == '__main__':
    test()
