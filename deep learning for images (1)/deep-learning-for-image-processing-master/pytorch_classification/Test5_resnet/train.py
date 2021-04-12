import os
import json #可以将json对象转化为普通的dict来处理

import torch
import torch.nn as nn
import torch.optim as  optim #优化器
from torchvision import transforms, datasets
from tqdm import tqdm #tqdm主要作用是用于显示进度，使用较为简单：
# 用户只需要封装任意的迭代器tqdm(iterator)即可完成进度条
from model import resnet34
import torchvision.models.resnet



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        #torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),#依据概率p对PIL图片进行垂直翻转
                                     transforms.ToTensor(),#将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]；
                                     # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        ##transforms.Normalize对图像标准化处理的参数，这里根据官网定义的
        #transforms.Normalize使用如下公式进行归一化：channel=（channel-mean）/std。前面三个是mean、后面三个是std
        #因为totensor已经转化为[0-1],故最后的结果范围是[-1~1]


        "val": transforms.Compose([transforms.Resize(256), #，长宽比固定不动，最小边缩放到256
                                   transforms.CenterCrop(224), #根据中心点剪切到224
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}#标准化处理，防止突出数值较高的指标在综合分析中的作用。

    data_root = os.path.abspath(os.path.join(os.getcwd(), "..\Test5_resnet"))  # 返回绝对路径.表示返回上上一层目录
    image_path = os.path.join(data_root, "data_set", "flower_data")# flower data set path
    # image_path = os.path.join(data_root, "data_set/flower_data") #再从根目录开始向下进行完整目录的拼接。
    # image_path = os.path.join("D:\pycharmproject\deep learning for images"
    #                                                       "\deep-learning-for-image-processing-master\pytorch_classification"
    #                                                       "\Test5_resnet/flower_data/data_set/flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    #断言（assert）作为一种软件调试的方法，提供了一种在代码中进行正确性检查的机制。
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    '''
    其构造函数如下：
 #ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，
ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
它主要有四个参数：
root：在root指定的路径下寻找图片
transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
target_transform：对label的转换
loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象
    '''
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx #类别名与数字类别的映射关系字典（class_to_idx）
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4) #json.dumps()是把python对象转换成json对象的一个过程，
    # 生成的是字符串。cla_dict是转化成json的对象，indent:参数根据数据格式缩进显示，读起来更加清晰。
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        '''
        因为传统打开文件是f = open('test.txt','r')如果存在则继续执行，然后f.close()；如果不存在则会报错
        改用with open as简化上面过程。无论有无异常均可自动调用f.close()方法。
        '''

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers 获取电脑的cpu个数。
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,#从实例化的dataset当中取得图片，
                                               # 然后打包成一个一个batch，然后输入网络进行训练
                                               batch_size=batch_size, shuffle=True,#打乱数据集
                                               num_workers=nw) #如果使用的是linux系统，
                                                # 将num_workers设置成一个大于0的数加快图像预处理进程
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),#在image-path中找到val文件夹
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth" #里面是预训练的权重
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False) #载入模型权重
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features #这里的fc是model.ResNet里的fc;in_features是输入特征矩阵的深度
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader) #tqdm显示进度
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
'''
运行结果：
train epoch[1/3] loss:0.206: 100%|██████████| 207/207 [27:59<00:00,  8.11s/it]
valid epoch[1/3]: 100%|██████████| 23/23 [03:17<00:00,  8.58s/it]
[epoch 1] train_loss: 0.473  val_accuracy: 0.852

train epoch[2/3] loss:1.016: 100%|██████████| 207/207 [23:06<00:00,  6.70s/it]
valid epoch[2/3]: 100%|██████████| 23/23 [01:34<00:00,  4.13s/it]
[epoch 2] train_loss: 0.339  val_accuracy: 0.901

train epoch[3/3] loss:0.636: 100%|██████████| 207/207 [22:05<00:00,  6.40s/it]
valid epoch[3/3]: 100%|██████████| 23/23 [01:07<00:00,  2.92s/it]
[epoch 3] train_loss: 0.278  val_accuracy: 0.915
Finished Training

'''