import torch.nn.functional as F
from torch import nn
from torchvision.models import VisionTransformer
import torch
class MatCNN(nn.Module):
    def __init__(self, hidden_size=64):
        super(MatCNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # 输入特征为 64*4*4
        self.fc2 = nn.Linear(512, 5)  # 输出类别数量
        # self.gelu = nn.GELU()


    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.pool(x)

        x = F.gelu(self.conv2(x))
        x = self.pool(x)

        x = F.gelu(self.conv3(x))
        x = self.pool(x)

        x = self.fc1(x)

        x = self.fc2(x)

        return x


class ImageRegressionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, embedding_dim, num_classes):
        super(ImageRegressionTransformer, self).__init__()

        # 定义用于提取图像patch的卷积层
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embedding_dim, kernel_size=patch_size,
                                         stride=patch_size)

        # 定义位置编码，这里是一个可学习的参数
        num_patches = (image_size // patch_size) ** 2
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

        # 定义类别标记(class token)，用于汇总所有patch的信息
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # 初始化VisionTransformer模型，这是一个预定义的Transformer模型
        # 注意这里的image_size和patch_size现在都指定了
        self.transformer = VisionTransformer(
            image_size=image_size,  # 输入图像的尺寸
            patch_size=patch_size,  # 图像划分成的patch的尺寸
            hidden_dim=embedding_dim,  # Transformer编码器中每个层的隐藏状态维度
            num_classes=num_classes,  # 模型输出的类别数量
            mlp_dim=embedding_dim * 4,  # MLP层的中间维度，通常为hidden_dim的4倍
            num_layers=num_layers,  # Transformer层数
            num_heads=num_heads,  # 注意力头的数量
            dropout=0.1,  # Dropout层的概率
            attention_dropout=0.1,  # Multi-Head Attention中的Dropout概率
        )

        # 定义回归头，将Transformer的输出映射到一个标量
        self.regression_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # 对输入图像执行patch embedding
        x = self.patch_embedding(x)  # 输入形状: B, C, H, W; 输出形状: B, E, H', W'

        # 展平并转置patch，准备输入Transformer
        x = x.flatten(2).transpose(1, 2)  # 输入形状: B, E, H'*W'; 输出形状: B, H'*W', E

        # 扩展类别标记以匹配批次大小
        class_token = self.class_token.expand(x.shape[0], -1, -1)

        # 在序列开头加入类别标记
        x = torch.cat([class_token, x], dim=1)

        # 添加位置编码
        x = x + self.positional_encoding[:, :x.size(1)]

        # 传递给Transformer模型
        x = self.transformer(x)

        # 从类别标记的输出获取回归预测
        x = self.regression_head(x[:, 0])

        # 返回回归预测值
        return x

# 初始化模型实例
# patch_size=16 表示每个patch的大小为16x16像素
# num_patches=196 表示一个输入图像被划分为196个patch
# embedding_dim=768 表示每个patch的嵌入维度
# num_heads=12 表示多头注意力机制的头数
# num_layers=12 表示Transformer编码器的层数
# num_classes=1 表示回归任务的输出维度为1
model = ImageRegressionTransformer(patch_size=16, num_patches=196, embedding_dim=768, num_heads=12, num_layers=12,
                                   num_classes=1)