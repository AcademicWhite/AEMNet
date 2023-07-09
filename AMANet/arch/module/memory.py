import torch
import torch.nn as nn

# 定义注意力机制
class AttentionModule(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(AttentionModule, self).__init__()

        self.query_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query = query.view(query.size(0), -1, query.size(2) * query.size(3)).permute(0, 2, 1)
        key = key.view(key.size(0), -1, key.size(2) * key.size(3))

        attention_map = torch.bmm(query, key)
        attention_map = self.softmax(attention_map)

        value = value.view(value.size(0), -1, value.size(2) * value.size(3))
        attention_value = torch.bmm(value, attention_map.permute(0, 2, 1))
        attention_value = attention_value.view(value.size(0), value.size(1), x.size(2), x.size(3))

        return attention_value

# 定义记忆模块
class MemoryModule(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(MemoryModule, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU()

        self.attention = AttentionModule(hidden_channels, hidden_channels)

    def forward(self, encoder_output):
        memory = self.conv1(encoder_output)
        memory = self.batchnorm1(memory)
        memory = self.relu1(memory)

        attention_memory = self.attention(memory)
        memory = memory + attention_memory

        return memory

# # 测试记忆模块
# encoder_output = torch.randn(5, 256, 32, 32)  # 假设输入大小为 (batch_size, channels, height, width)
# memory_module = MemoryModule(input_channels=256, hidden_channels=256)
# memory = memory_module(encoder_output)
# # print(memory.shape)