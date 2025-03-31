import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

#定义策略网络
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    # 前向传播
    def forward(self, x):  #
        x = torch.relu(self.fc1(x)) #
        x = torch.relu(self.fc2(x))  #
        return self.fc3(x) #输出动作分数

def load_model(model_path="C:/Users/28123/Desktop/model.pth"):
    model = PolicyNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    print(f"Model loaded from {model_path}")
    return model

def test_policy(policy_net, steps=1000):
    state = torch.tensor([[np.random.rand() * 2]], dtype=torch.float32)  # 随机初始化状态
    search_path = [state.item()]  # 记录搜索路径

    for _ in range(steps):
        scores = policy_net(state)  # 获取策略网络的分数
        probabilities = torch.nn.functional.softmax(scores, dim=1)  # 计算动作概率
        action = torch.argmax(probabilities).item()  # 选择概率最高的动作

        step_size = 0.01
        new_state = state - step_size if action == 0 else state + step_size  # 更新状态
        state = new_state  # 赋值新的状态
        search_path.append(state.item())  # 记录路径

    return search_path

def plot_search_path(search_path):
    plt.figure(figsize=(8, 5))
    plt.plot(search_path, marker="o", linestyle="-", color="b", alpha=0.7, label="Search Path")
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.title("Policy Search Path")
    plt.legend()
    plt.grid(True)
    plt.show()

trained_model = load_model("C:/Users/28123/Desktop/policy_net.pth")
search_path = test_policy(trained_model, steps=100)
plot_search_path(search_path)
