# CNN
EDA二轮考核

代码框架包括：
  1.数据加载函数
  2.数据预处理函数
  3.数据集划分函数
  4.CNN模型定义
  5.训练函数
  6.测试函数
  7.主函数

---

### 代码调用了以下库：

1. **`numpy`**：
   - 用于科学计算，提供高效的数组操作和数学函数。
   - 在代码中可能用于数据处理和矩阵运算。

2. **`torch`**：
   - PyTorch 是深度学习框架，提供了张量操作、自动求导和神经网络模块。
   - 在代码中用于定义和训练模型。

3. **`torch.nn`**：
   - PyTorch 的神经网络模块，提供了各种层（如卷积层、全连接层）和损失函数。
   - 在代码中用于定义 CNN 模型和损失函数。

4. **`torch.nn.functional`**：
   - PyTorch 的函数模块，提供了各种激活函数（如 ReLU）和损失函数。
   - 在代码中可能用于激活函数和损失计算。

5. **`torch.utils.data`**：
   - PyTorch 的数据加载模块，提供了 `DataLoader`、`TensorDataset` 和 `random_split` 等工具。
   - 在代码中用于加载和划分数据集。

6. **`sklearn.model_selection`**：
   - Scikit-learn 的模型选择模块，提供了 `train_test_split` 函数。
   - 在代码中用于划分训练集、验证集和测试集。

7. **`time`**：
   - Python 标准库，提供了时间相关函数。
   - 在代码中用于记录训练时间。

8. **`psutil`**：
   - 用于监控系统和进程的资源使用情况（如 CPU、内存）。
   - 在代码中用于记录训练过程中的内存占用。

---

### 环境设置：

pytorch环境（已安装pytorch geometric），cuda

---

### 输出结果：

在分支run_output中查看

https://github.com/zhuang3346/CNN/tree/run_output

