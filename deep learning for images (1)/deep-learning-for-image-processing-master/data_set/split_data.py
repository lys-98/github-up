import os
from shutil import copy, rmtree
#shutil模块提供了大量的文件的高级操作。特别针对文件拷贝和删除，
# 主要功能为目录和文件操作以及压缩操作。对单个文件的操作也可参见os模块。
#copy复制一个文件到一个文件或一个目录
#rmtree递归删除一个目录以及目录内的所有内容
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹再重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现。使得随机数据可预测，即只要seed的值一样，后续生成的随机数都一样
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 指向你解压后的flower_photos文件夹
    '''
    os.getcwd()
    该函数不需要传递参数，它返回当前的目录。需要说明的是，
    当前目录并不是指脚本所在的目录，而是所运行脚本的目录。
    
    ../”来表示上一级目录，“../../”表示上上级的目录, /表示从根目录开始，./表示从当前目录开始
    '''
    # cwd = os.getcwd()
    # data_root = os.path.join(cwd, "flower_data")
    data_root = os.path.abspath(os.path.join(os.getcwd(), "..\data_set"))
    origin_flower_path = os.path.join(data_root, "flower_photos")
    assert os.path.exists(origin_flower_path)
    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]
    #os.listdir()用于返回一个由文件名和目录名组成的列表，需要注意的是它接收的参数需要是一个绝对的路径
    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))#random.sample（）在images里面随机选出k个数据
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
