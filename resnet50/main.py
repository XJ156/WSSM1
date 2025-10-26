import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# https://blog.csdn.net/zhaohongfei_358/article/details/122742656


# 五、训练
from net import ResModel
import net

def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_FN = 0
        train_FP = 0
        train_TP = 0
        train_TN = 0
        valid_FN = 0
        valid_FP = 0
        valid_TP = 0
        valid_TN = 0
        # 照片读取上来的维度： [通道数，长，宽]
        for k, res_list in enumerate(tqdm(train_data)):
            inputs = res_list[0].to(device)
            #print(inputs['image_name'])

            labels = res_list[1].to(device)#标签，系统自动处理

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            for i in range(len(predictions)):
                if predictions[i] == labels[i] and labels[i] == 0:
                    train_TP += 1
                elif predictions[i] == labels[i] and labels[i] == 1:
                    train_TN += 1
                elif predictions[i] == 1 and labels[i] == 0:
                    train_FN += 1
                elif predictions[i] == 0 and labels[i] == 1:
                    train_FP += 1
        with torch.no_grad():
            model.eval()

            for j, rs_list in enumerate(tqdm(valid_data)):
                inputs = rs_list[0].to(device)
                labels = rs_list[1].to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)
 

                for i in range(len(predictions)):
                    if predictions[i] == labels[i] and labels[i] == 0:
                        valid_TP += 1
                    elif predictions[i] == labels[i] and labels[i] == 1:
                        valid_TN += 1
                    elif predictions[i] == 1 and labels[i] == 0:
                        valid_FN += 1
                    elif predictions[i] == 0 and labels[i] == 1:
                        valid_FP += 1

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            best_TP=valid_TP
            best_FP=valid_FP
            best_TN=valid_TN
            best_FN=valid_FN

      

            #torch.save(model.state_dict(), 'OnlyChangModel/' + dataset + '_model_' + str(epoch + 1) + '.pth')
        epoch_end = time.time()
        #打印loss和准确度
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100,  avg_valid_loss, avg_valid_acc * 100, 
                epoch_end - epoch_start
            ))

        print("Epoch:{:03d}, train_data_size: {:03d}, valid_data_size:{:03d}, \n\t\t "
              "train_TP: {:03d}, train_TN: {:03d}, train_FP: {:03d}, train_FN: {:03d},\n\t\t"
              "valid_TP: {:03d}, valid_TN: {:03d}, valid_FP: {:03d}, valid_FN: {:03d},".format(epoch + 1,
                                                                                               train_data_size,
                                                                                               valid_data_size,
                                                                                               train_TP, train_TN,
                                                                                               train_FP, train_FN,
                                                                                               valid_TP, valid_TN,
                                                                                               valid_FP, valid_FN))

        print("Best Accuracy for validation : {:.4f} ;      Best TP for validation : {:.1f};Best FP for validation : {:.1f};     Best TN for validation : {:.1f};        Best FN for validation : {:.1f};    at epoch {:03d}".format(best_acc, best_TP, best_FP,best_TN ,best_FN,best_epoch))

        if (epoch > 290):
            torch.save(model.state_dict(), '/disk/sdb/zhangqian0620/fenlei/save/'  + str(epoch + 1) + '.pth')

    return model, history


if __name__ == '__main__':
    # 一、建立数据集
    # 二、数据增强
    # 建好的数据集在输入网络之前先进行数据增强，包括随机 resize 裁剪到 256 x 256，随机旋转，随机水平翻转，中心裁剪到 224 x 224，转化成 Tensor，正规化等。
    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪图像，长宽为 256*256
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # 三、加载数据
    # torchvision.transforms包DataLoader是 Pytorch 重要的特性，它们使得数据增加和加载数据变得非常简单。
    # 使用 DataLoader 加载数据的时候就会将之前定义的数据 transform 就会应用的数据上了。

    dataset = '/disk/sdb/zhangqian0620/fenlei/data'
    # pretrained_file = "model/resnet50-5c106cde.pth"

    # isLocal = False
    # if isLocal:
    #     datasets = "../" + str(datasets)
    #     pretrained_file = "../" + str(pretrained_file)



    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')

    batch_size = 8
   #对数据进行一个封装
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }

    train_data_size = len(data['train'])
    # print(train_data_size)
    valid_data_size = len(data['valid'])
    #print(data['train'])
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=4)

    # 四、迁移学习
    # 这里使用ResNet-50的预训练模型。

    #因为是2分类问题所以写的2
    resnet50 = ResModel(2)
    #resnet50=net.SimCLRStage2(num_class=2)
    # resnet50.load_state_dict(torch.load(pretrained_file))

    # resnet50 = models.resnet50(pretrained=True)

    # 在PyTorch中加载模型时，所有参数的‘requires_grad’字段默认设置为true。这意味着对参数值的每一次更改都将被存储，以便在用于训练的反向传播图中使用。
    # 这增加了内存需求。由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。

    # for param in resnet50.parameters():
    #     param.requires_grad = False

    # 为了适应自己的数据集，将ResNet-50的最后一层替换为，
    # 将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，接着再连接ReLU层和Dropout层，然后是256 x 6的线性层，输出为6通道的softmax层。
    # fc_inputs = resnet50.fc.in_features
    # resnet50.fc = nn.Sequential(
    #     nn.Linear(fc_inputs, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, 2),
    #     nn.LogSoftmax(dim=1)
    # )

    # 用GPU进行训练。
    if torch.cuda.is_available():
        resnet50 = resnet50.to('cuda:0')
        print("*" * 20, "使用了GPU！")
    else:
        print("*" * 20, "使用了CPU！")
    # 定义损失函数和优化器。
   # loss_func = nn.NLLLoss()
    loss_func=nn.CrossEntropyLoss()
    #loss_func=net.Loss()
    optimizer = optim.Adam(resnet50.parameters())

   # optimizer = optim.SGD(resnet50.parameters())
    num_epochs =300
    #跳转到train_and_valid方法
    trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
    print(trained_model)
    torch.save(history, 'models/' + '_history.pt')

    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_loss_curve.png')

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_accuracy_curve.png')
