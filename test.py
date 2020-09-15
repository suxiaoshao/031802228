import os

import main


def test():
    copy_paths = './copy'
    origin_path = './origin.txt'
    for path in os.listdir(copy_paths):
        print(f'{path}:', main.get_similarity(f'{copy_paths}/{path}', origin_path))


if __name__ == '__main__':
    test()
