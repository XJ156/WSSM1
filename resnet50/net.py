import torch

from torch import nn
class ConvBlock(nn.Module):
    # 每个stage第一个卷积模块，主要进行下采样
    def __init__(self, in_channel, f, filters, s):
        # resnet50 只有 131卷积
        # filters传入的是一个 [filter1, filter2, filter3] 列表
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=F1, kernel_size=1, stride=s, padding=0, bias=False),  # 1*1卷积
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=F1, out_channels=F2,kernel_size=f, stride=1, padding=1, bias=False),  # 3*3 卷积
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0, bias=False),  # 1*1 卷积
            nn.BatchNorm2d(F3),
        )

        # self.c1 = nn.Conv2d(in_channels=in_channel, out_channels=F1, kernel_size=1, stride=s, padding=0, bias=False)  # 1*1卷积
        # self.b1 = nn.BatchNorm2d(F1)
        # self.r1 = nn.ReLU(True)
        # self.c2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=1, padding=0, bias=False)  # 3*3 卷积
        # self.b2 = nn.BatchNorm2d(F2)
        # self.r2 = nn.ReLU(True)
        # self.c3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0, bias=False)  # 1*1 卷积
        # self.b3 = nn.BatchNorm2d(F3)



        # 短路部分，从输入in_channel直接卷积到输出F3，恒等映射
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)



        X = self.stage(X)
        # X = self.c1(X)
        # X = self.b1(X)
        # #X = self.r1(X)
        # X = self.c2(X)
        # X = self.b2(X)
        # #X = self.r2(X)
        # X = self.c3(X)
        # X = self.b3(X)





        # 残差加和
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


# stage内部的卷积模块
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

'''
ResNet核心
stage1: 1层卷积，1层池化
stage2: 三个131卷积(1*1, 3*3, 1*1), 共9层卷积
stage3: 四个131卷积，共12层卷积
stage4: 六个131卷积，共18层卷积
stage5: 三个131卷积，9层
fc: 全连接

conv2d = 1+9+12+18+9+1 = 50
'''
class ResModel(nn.Module):
    '''
    resnet50：包含了50个conv2d操作，共分5个stage

    stage1: 1层卷积，1层池化
    stage2: 三个131卷积(1*1, 3*3, 1*1), 共9层卷积
    stage3: 四个131卷积，共12层卷积
    stage4: 六个131卷积，共18层卷积
    stage5: 三个131卷积，9层
    fc: 全连接

    conv2d = 1+9+12+18+9+1 = 50
    '''

    def __init__(self, n_class=2):
        '''
        ConvBlock相比IndentityBlock，多了使用stride下采样的环节，且Identity不进行bn

        所以每个stage都是通过ConvBlock进入，决定是否下采样
        '''
        super(ResModel, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # 3通道输入，64个7*7卷积，步长2
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )  # 3个卷积

        self.stage3 = nn.Sequential(
            ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )  # 4个卷积块

        self.stage4 = nn.Sequential(
            ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
        )  # 6个卷积块

        self.stage5 = nn.Sequential(
            ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
            IndentityBlock(2048, 3, [512, 512, 2048]),
            IndentityBlock(2048, 3, [512, 512, 2048]),
        )  # 3个卷积块

        # 平均池化
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            #nn.Linear(32768, n_class)  # 输入参数等于 = 2048 * 4 * 4， 4是最后一层池化输出大小
            nn.Linear(32768, 4096),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        # 输出的尺寸是 [32, 2048, 4, 4]
        out = out.view(X.shape[0], -1)
        # 调整后 [32, 32768]
        out = self.fc(out)


        return out
