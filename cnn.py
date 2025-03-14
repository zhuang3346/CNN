import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import time
import psutil


# 数据加载函数
def loadCoraData(filePath):
    """
    加载Cora数据集的特征和标签。
    参数:
        filePath (str): cora.content文件路径
    返回:
        features (np.ndarray): 特征矩阵，形状为(num_nodes, num_features)
        labels (np.ndarray): 标签向量，形状为(num_nodes,)
    """
    data = []   # 用于存储节点特征
    labels = []     # 用于存储节点标签
    labelMap = {}  # 一部字典，用于将类别标签映射为整数
    currentLabelId = 0  # 当前标签的整数ID

#cora.content文件格式：每一行表示一个节点<node_id> <feature_1> <feature_2> ... <feature_n> <label>

    with open(filePath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # 特征部分（1433维）
            features = list(map(int, parts[1:-1]))
            # 标签部分
            labelStr = parts[-1]
            if labelStr not in labelMap:    #每次出现新标签时进行对labelMap字典的写入
                labelMap[labelStr] = currentLabelId
                currentLabelId += 1
            labels.append(labelMap[labelStr])   #从labelMap字典中查找当前节点的整数编号，并将其填入labels列表中
            data.append(features)

    return np.array(data, dtype=np.float32), np.array(labels)
    #返回两个量，data为特征矩阵，labels为标签向量


# 数据预处理函数
def preprocessData(features, targetHeight=38, targetWidth=38):
    """
    将每个节点的特征填充并转换为伪图像格式。
    参数:
        features (np.ndarray): 原始特征矩阵，形状为(num_nodes, num_features)
        targetHeight (int): 目标图像高度
        targetWidth (int): 目标图像宽度
    返回:
        images (np.ndarray): 转换后的CNN可接受的4D张量图像数据，形状为(num_nodes, 1, H, W)，其中1是通道数
    """
    numNodes, numFeatures = features.shape  # 获取节点数量和特征数量
    targetSize = targetHeight * targetWidth # 计算目标图像尺寸
    padSize = targetSize - numFeatures      # 计算需要填充的零的数量，padSide是填充的大小
    if padSize < 0:
        raise ValueError("目标尺寸小于特征数量，请调整targetHeight和targetWidth")

    paddedFeatures = np.zeros((numNodes, targetSize), dtype=np.float32) # 填充零
    paddedFeatures[:, :numFeatures] = features  # 将原始特征复制到被零填充后的数组的前numFeatures列中

    # 重塑为图像格式 (通道数=1)
    images = paddedFeatures.reshape(numNodes, 1, targetHeight, targetWidth)
    return images


# 数据集划分函数
def splitDataset(images, labels, testRatio=0.1, valRatio=0.1):
    """
    按 8：1：1 划分数据集为训练集、验证集、测试集。
    参数:
        images (np.ndarray): 图像数据，形状为(num_nodes, 1, H, W)
        labels (np.ndarray): 标签数据
        testRatio (float): 测试集比例
        valRatio (float): 验证集比例（从剩余数据中划分）
    返回:
        划分后的六个数据集 (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 先划分训练+验证和测试集
    X_trainVal, X_test, y_trainVal, y_test = train_test_split(
        images, labels, test_size=testRatio, stratify=labels, random_state=42
    )
    # 再从训练+验证中划分验证集
    valRatioAdjusted = valRatio / (1 - testRatio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainVal, y_trainVal, test_size=valRatioAdjusted, stratify=y_trainVal, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# CNN模型定义
class CNNModel(nn.Module):
    """
        定义一个2层CNN模型，用于图分类任务。
        输入:
            inputChannels (int): 输入通道数，默认为1（单通道图像）
            numClasses (int): 输出类别数，默认为7
        输出:
            x (torch.Tensor): 模型的输出张量，形状为(batch_size, numClasses)
    """
    def __init__(self, inputChannels=1, numClasses=7):
        super(CNNModel, self).__init__()
        # 第一层卷积，输出通道数为32，卷积核大小为3
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=3)
        # 最大池化层，核大小为2，步长为2
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积，输出通道数为64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 计算全连接层输入尺寸（假设输入图像为38x38）
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # 输出层
        self.fc2 = nn.Linear(128, numClasses)

    def forward(self, x):   # 输入x=（batch_size(即num_nodes), 1, 38, 38）
        # 第一层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 输出形状: (num_nodes（后续几个注释省略此项）, 32, 18, 18)
        # 第二层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 输出形状: (64, 8, 8)
        # 展平
        x = x.view(-1, 64 * 8 * 8)
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x


# 训练函数
def trainModel(model, trainLoader, valLoader, numEpochs=20, learningRate=0.001):
    """
        训练CNN模型。
        输入:
            model (nn.Module): 待训练的CNN模型
            trainLoader (DataLoader): 训练集数据加载器
            valLoader (DataLoader): 验证集数据加载器
            numEpochs (int): 训练轮次，默认为20
            learningRate (float): 学习率，默认为0.001
        输出:
            trainingTime (float): 训练总时间（秒）
            memoryUsage (int): 训练过程中内存占用的增量（MB）
    """
    #初始化优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # 记录训练过程中的训练时间（开始计时）和内存占用
    startTime = time.time()
    process = psutil.Process()
    startMem = process.memory_info().rss // 1024 // 1024  # 初始内存（MB）

    for epoch in range(numEpochs):  # numEpochs为训练批次
        model.train()
        runningLoss = 0.0
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)  # 训练数据迁移到GPU
            optimizer.zero_grad()   # 梯度清零
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()         # 反向传播
            optimizer.step()        # 参数更新
            runningLoss += loss.item()  # 累计当前 epoch 中所有批次(Batch)的损失值

        # 验证集评估
        model.eval()    # 将模型转为（验证）评估模式
        valCorrect = 0
        valTotal = 0
        with torch.no_grad():   # 关闭梯度计算，节省内存
            for inputs, labels in valLoader:
                inputs, labels = inputs.to(device), labels.to(device)  # 验证数据迁移到GPU
                outputs = model(inputs)     # 前向传播
                _, predicted = torch.max(outputs.data, 1)   # 获取预测结果，即输出概率最大值的索引（忽略第一项，获取第二项）
                valTotal += labels.size(0)  # 累计验证集总样本数量
                valCorrect += (predicted == labels).sum().item()    # 累计正确预测的数量
        valAcc = valCorrect / valTotal
        print(f'Epoch {epoch + 1}/{numEpochs}, Loss: {runningLoss:.3f}, Val Acc: {valAcc:.3f}')

    endTime = time.time()
    endMem = process.memory_info().rss // 1024 // 1024
    trainingTime = endTime - startTime
    memoryUsage = endMem - startMem
    return trainingTime, memoryUsage


# 测试函数
def evaluateModel(model, testLoader):
    """
        在测试集上评估CNN模型的分类准确率。
        输入:
            model (nn.Module): 训练好的CNN模型
            testLoader (DataLoader): 测试集数据加载器
        输出:
            accuracy (float): 模型在测试集上的分类准确率
    """
    model.eval()    # 将模型转为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model(inputs)    # 前向传播
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # 累计测试集总样本数量
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# 主函数
if __name__ == '__main__':
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载数据
    features, labels = loadCoraData('cora.content')
    # 预处理为伪图像 (38x38)
    images = preprocessData(features, targetHeight=38, targetWidth=38)
    # 按8：1：1划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = splitDataset(
        images, labels, testRatio=0.1, valRatio=0.1
    )
    # 将数据集转换为Tensor
    trainDataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valDataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    testDataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # 创建DataLoader
    batchSize = 32
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize)
    testLoader = DataLoader(testDataset, batch_size=batchSize)
    # 初始化模型
    model = CNNModel(inputChannels=1, numClasses=7).to(device)  # 将模型加载到GPU
    # 训练模型
    trainingTime, memoryUsage = trainModel(model, trainLoader, valLoader, numEpochs=100)
    # 测试评估
    testAccuracy = evaluateModel(model, testLoader)
    print(f'Test Accuracy: {testAccuracy:.4f}')
    print(f'Training Time: {trainingTime:.2f} seconds')
    print(f'Memory Usage: {memoryUsage} MB')