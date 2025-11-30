import numpy as np
from typing import Tuple, Optional, Dict
import time


class ConvLayer:
    """卷积层 - 使用im2col加速"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier初始化 (适用于Sigmoid激活函数)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size + out_channels))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)
        
        # 缓存用于反向传播
        self.x_col = None
        self.x_padded = None
    
    def im2col(self, x: np.ndarray) -> np.ndarray:
        """
        im2col操作，将多维图像转换为2D矩阵以加速卷积
        
        Args:
            x: 输入图像 (batch_size, in_channels, height, width)
            
        Returns:
            col_matrix: 转换后的矩阵
        """
        batch_size, in_channels, height, width = x.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), 
                                 (self.padding, self.padding), 
                                 (self.padding, self.padding)), 
                             mode='constant')
        else:
            x_padded = x
        
        # 存储用于反向传播
        self.x_padded = x_padded
        
        # 创建col矩阵
        col_matrix = np.zeros((batch_size, in_channels, self.kernel_size, self.kernel_size,
                               out_height, out_width))
        
        # 提取所有局部块
        for i in range(self.kernel_size):
            i_max = i + self.stride * out_height
            for j in range(self.kernel_size):
                j_max = j + self.stride * out_width
                col_matrix[:, :, i, j, :, :] = x_padded[:, :, i:i_max:self.stride, j:j_max:self.stride]
        
        # 重塑为 (batch_size * out_height * out_width, in_channels * kernel_size * kernel_size)
        col_matrix = col_matrix.transpose(0, 4, 5, 1, 2, 3).reshape(
            batch_size * out_height * out_width, -1)
        
        self.x_col = col_matrix
        return col_matrix
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入 (batch_size, in_channels, height, width)
            
        Returns:
            output: 卷积结果 (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape
        
        # im2col转换
        x_col = self.im2col(x)
        
        # 重塑权重为 (out_channels, in_channels * kernel_size * kernel_size)
        w_col = self.weights.reshape(self.out_channels, -1)
        
        # 矩阵乘法计算卷积
        output = np.dot(x_col, w_col.T) + self.bias
        
        # 重塑回4D张量
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = output.reshape(batch_size, out_height, out_width, self.out_channels).transpose(0, 3, 1, 2)
        
        return output
    
    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        反向传播
        
        Args:
            dout: 上一层的梯度 (batch_size, out_channels, out_height, out_width)
            learning_rate: 学习率
            
        Returns:
            dx: 对输入的梯度
        """
        batch_size, out_channels, out_height, out_width = dout.shape
        
        # 重塑 dout 为 (batch_size * out_height * out_width, out_channels)
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(batch_size * out_height * out_width, -1)
        
        # 计算权重梯度
        dw = np.dot(dout_reshaped.T, self.x_col)
        dw = dw.reshape(self.weights.shape)
        
        # 计算偏置梯度
        db = np.sum(dout_reshaped, axis=0)
        
        # 更新参数
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        # 计算输入梯度
        w_reshaped = self.weights.reshape(self.out_channels, -1)
        dx_col = np.dot(dout_reshaped, w_reshaped)
        
        # 将dx_col重塑回4D张量 (col2im操作)
        dx = self.col2im(dx_col, batch_size, out_height, out_width)
        
        return dx
    
    def col2im(self, col_matrix: np.ndarray, batch_size: int, 
               out_height: int, out_width: int) -> np.ndarray:
        """col2im操作，将2D矩阵转换回图像格式"""
        in_channels, height, width = self.in_channels, self.x_padded.shape[2], self.x_padded.shape[3]
        
        # 重塑col_matrix
        col_matrix = col_matrix.reshape(batch_size, out_height, out_width, 
                                       in_channels, self.kernel_size, self.kernel_size)
        col_matrix = col_matrix.transpose(0, 3, 4, 5, 1, 2)
        
        # 初始化输出
        dx_padded = np.zeros_like(self.x_padded)
        
        # 累加梯度
        for i in range(self.kernel_size):
            i_max = i + self.stride * out_height
            for j in range(self.kernel_size):
                j_max = j + self.stride * out_width
                dx_padded[:, :, i:i_max:self.stride, j:j_max:self.stride] += col_matrix[:, :, i, j, :, :]
        
        # 移除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx


class AvgPoolLayer:
    """平均池化层"""
    
    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        self.pool_size = pool_size
        self.stride = stride or pool_size
        
        # 缓存
        self.x_shape = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入 (batch_size, channels, height, width)
            
        Returns:
            output: 池化结果
        """
        self.x_shape = x.shape
        batch_size, channels, height, width = x.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                output[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        Args:
            dout: 上一层的梯度
            
        Returns:
            dx: 对输入的梯度
        """
        batch_size, channels, out_height, out_width = dout.shape
        _, _, height, width = self.x_shape
        
        dx = np.zeros(self.x_shape)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # 平均分配梯度
                dx[:, :, h_start:h_end, w_start:w_end] += dout[:, :, i:i+1, j:j+1] / (self.pool_size * self.pool_size)
        
        return dx


class FCLayer:
    """全连接层"""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier初始化
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)
        
        # 缓存
        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入 (batch_size, in_features)
            
        Returns:
            output: 全连接结果 (batch_size, out_features)
        """
        self.x = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        反向传播
        
        Args:
            dout: 上一层的梯度
            learning_rate: 学习率
            
        Returns:
            dx: 对输入的梯度
        """
        # 计算梯度
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.weights.T)
        
        # 更新参数
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return dx


class Sigmoid:
    """Sigmoid激活函数"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x: np.ndarray, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        return dout * sigmoid_x * (1 - sigmoid_x)


class Softmax:
    """Softmax激活函数"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 数值稳定性
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """反向传播 (Cross-Entropy Loss的梯度)"""
        return y_pred - y_true


class LeNet5NumPy:
    """LeNet-5的纯NumPy实现（适配MNIST）"""
    
    def __init__(self):
        """初始化网络层"""
        # 构建网络 - 适配MNIST 28x28输入
        self.conv1 = ConvLayer(1, 6, 5, stride=1, padding=0)    # C1: 6@5x5 -> 24x24
        self.pool2 = AvgPoolLayer(2, stride=2)                   # S2: 2x2平均池化 -> 12x12
        self.conv3 = ConvLayer(6, 16, 5, stride=1, padding=0)   # C3: 16@5x5 -> 8x8
        self.pool4 = AvgPoolLayer(2, stride=2)                   # S4: 2x2平均池化 -> 4x4
        # 改为：直接将4x4x16 flatten，不使用5x5卷积（会导致0x0输出）
        # 这样更接近原始论文中对32x32输入的设计（经过处理后也是4x4）
        self.fc5 = FCLayer(16 * 4 * 4, 120)                    # F5: 120
        self.fc6 = FCLayer(120, 84)                            # F6: 84
        self.output = FCLayer(84, 10)                           # Output: 10
        
        # 激活函数
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        
        # 损失历史
        self.loss_history = []
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入 (batch_size, 1, 28, 28)
            training: 是否为训练模式
            
        Returns:
            output: 网络输出 (batch_size, 10)
        """
        # 缓存中间结果用于反向传播
        if training:
            self.cache = {}
        
        # C1层
        conv1_out = self.conv1.forward(x)
        conv1_activated = self.sigmoid.forward(conv1_out)
        if training:
            self.cache['conv1_out'] = conv1_out
            self.cache['conv1_activated'] = conv1_activated
        
        # S2层
        pool2_out = self.pool2.forward(conv1_activated)
        if training:
            self.cache['pool2_out'] = pool2_out
        
        # C3层
        conv3_out = self.conv3.forward(pool2_out)
        conv3_activated = self.sigmoid.forward(conv3_out)
        if training:
            self.cache['conv3_out'] = conv3_out
            self.cache['conv3_activated'] = conv3_activated
        
        # S4层
        pool4_out = self.pool4.forward(conv3_activated)
        if training:
            self.cache['pool4_out'] = pool4_out
        
        # Flatten - 直接flatten为FC层输入
        flattened = pool4_out.reshape(pool4_out.shape[0], -1)
        if training:
            self.cache['flattened'] = flattened
        
        # F5层
        fc5_out = self.fc5.forward(flattened)
        fc5_activated = self.sigmoid.forward(fc5_out)
        if training:
            self.cache['fc5_out'] = fc5_out
            self.cache['fc5_activated'] = fc5_activated
        
        # F6层
        fc6_out = self.fc6.forward(fc5_activated)
        fc6_activated = self.sigmoid.forward(fc6_out)
        if training:
            self.cache['fc6_out'] = fc6_out
            self.cache['fc6_activated'] = fc6_activated
        
        # Output层
        output_raw = self.output.forward(fc6_activated)
        output = self.softmax.forward(output_raw)
        
        return output
    
    def backward(self, y_true: np.ndarray, learning_rate: float) -> float:
        """
        反向传播
        
        Args:
            y_true: 真实标签 (batch_size, 10)
            learning_rate: 学习率
            
        Returns:
            loss: 当前批次的损失
        """
        # 计算损失
        y_pred = self.softmax.forward(self.cache['fc6_activated'] @ self.output.weights + self.output.bias)
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        self.loss_history.append(loss)
        
        # Output层反向传播
        dout = self.softmax.backward(y_pred, y_true)
        d_fc6 = self.output.backward(dout, learning_rate)
        
        # F6层反向传播
        d_fc6_raw = self.sigmoid.backward(self.cache['fc6_out'], d_fc6)
        d_fc5_activated = self.fc6.backward(d_fc6_raw, learning_rate)
        
        # F5层反向传播
        d_fc5_raw = self.sigmoid.backward(self.cache['fc5_out'], d_fc5_activated)
        d_flattened = self.fc5.backward(d_fc5_raw, learning_rate)
        
        # 重塑为卷积层的形状
        d_pool4_activated = d_flattened.reshape(self.cache['pool4_out'].shape)
        
        # S4层反向传播
        d_conv3_activated = self.pool4.backward(d_pool4_activated)
        
        # C3层反向传播
        d_conv3_raw = self.sigmoid.backward(self.cache['conv3_out'], d_conv3_activated)
        d_pool2 = self.conv3.backward(d_conv3_raw, learning_rate)
        
        # S2层反向传播
        d_conv1_activated = self.pool2.backward(d_pool2)
        
        # C1层反向传播
        d_conv1_raw = self.sigmoid.backward(self.cache['conv1_out'], d_conv1_activated)
        _ = self.conv1.backward(d_conv1_raw, learning_rate)
        
        return loss
    
    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """单步训练"""
        y_pred = self.forward(x, training=True)
        loss = self.backward(y, learning_rate)
        return loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测"""
        y_pred = self.forward(x, training=False)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        y_pred = self.forward(x, training=False)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        
        accuracy = np.mean(y_pred_labels == y_true_labels)
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
        
        return accuracy, loss
    
    def get_all_conv_weights(self) -> Dict[str, np.ndarray]:
        """获取所有卷积层的权重用于可视化"""
        return {
            'Conv1': self.conv1.weights.copy(),
            'Conv3': self.conv3.weights.copy()
        }
    
    def get_first_layer_weights(self) -> np.ndarray:
        """获取第一层卷积核用于可视化（保持向后兼容）"""
        return self.conv1.weights.copy()


if __name__ == "__main__":
    # 测试LeNet-5
    model = LeNet5NumPy()
    
    # 创建随机测试数据
    batch_size = 4
    x = np.random.randn(batch_size, 1, 28, 28)
    y = np.random.randint(0, 10, (batch_size, 10))
    y = np.eye(10)[y.argmax(axis=1)]
    
    print(f"输入形状: {x.shape}")
    print(f"标签形状: {y.shape}")
    
    # 前向传播测试
    start_time = time.time()
    output = model.forward(x)
    forward_time = time.time() - start_time
    
    print(f"输出形状: {output.shape}")
    print(f"前向传播时间: {forward_time:.4f}s")
    
    # 训练步骤测试
    start_time = time.time()
    loss = model.train_step(x, y, learning_rate=0.01)
    backward_time = time.time() - start_time
    
    print(f"损失: {loss:.4f}")
    print(f"反向传播时间: {backward_time:.4f}s")
    
    # 评估测试
    accuracy, eval_loss = model.evaluate(x, y)
    print(f"准确率: {accuracy:.4f}")
    print(f"评估损失: {eval_loss:.4f}")
