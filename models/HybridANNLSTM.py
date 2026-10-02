import torch
import torch.nn as nn

# LSTM 模块
class TemporalStream(nn.Module):
    def __init__(self, input_channel=10, output_channel=64):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.lstm = nn.LSTM(input_size=self.input_channel, hidden_size=self.output_channel, num_layers=1, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 只取最后一个时间步的隐藏状态
        return lstm_out[:, -1, :]
    
    
# ANN 模块
class SpatialStream(nn.Module):
    def __init__(self, input_channel=10, output_channel=64, dropout=0.2):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.dropout = dropout

        # 两层 MLP
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.input_channel, out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=self.output_channel),
            nn.BatchNorm1d(self.output_channel),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        # 只取最后一天的特征
        input_features = x[:, -1, :]

        return self.layer2(self.layer1(input_features))
    


# 特征融合与预测头
class FusionAndPredictHead(nn.Module):
    def __init__(self, input_channel=128, output_channel=1, dropout=0.2):
        super().__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.dropout = dropout
            
        self.head = nn.Sequential(
            nn.Linear(in_features=input_channel, out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=32, out_features=output_channel)
        )
        
    def forward(self, x):
        # x: [Batch_size, 128]
        pred = self.head(x)

        return pred
    


# 整合模型
class HybridANNLSTM(nn.Module):
    def __init__(self, n_features=10, lstm_hidden_dim=64, ann_hidden_dim=64, dropout=0.2):
        super().__init__()
        # 实例化各个模块
        self.temporal_stream = TemporalStream(n_features, lstm_hidden_dim)
        self.spatial_stream = SpatialStream(n_features, ann_hidden_dim, dropout)
        self.fusion_head = FusionAndPredictHead(lstm_hidden_dim + ann_hidden_dim, 1, dropout)

    def forward(self, x):
        # x 形状：[Batch_size, T=30, 10]
        out_lstm = self.temporal_stream(x)
        out_ann = self.spatial_stream(x)

        cat_input = torch.cat((out_lstm, out_ann), dim=-1)
        
        pred = self.fusion_head(cat_input)

        return pred
    

if __name__ == "__main__":
    # 模拟生成一个 Batch 为 2，观察窗口 30 天，10个特征的 Tensor
    mock_input = torch.randn(2, 30, 10)
    
    # 实例化的主模型
    model = HybridANNLSTM()
    
    # 前向传播
    output = model(mock_input)
    print("模型运行成功！预测输出形状为:", output.shape) # 预期输出: torch.Size([2, 1])