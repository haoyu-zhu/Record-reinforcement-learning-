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


def parabola(x):
    return (x-1)**2 + 1

policy_net = PolicyNet()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)

def train_PolicyNet(epochs=100, steps_per_epoch=100,epsilon=0.1,min_loss=0.001, model_path=""):
    search_path = []

    for epoch in range(epochs):
        state = torch.tensor([[np.random.rand() * 2]],dtype=torch.float32)  # 初始化状
        #state = torch.tensor(5,dtype=torch.float32)
        optimizer.zero_grad() #梯度清零
        log_probs =[] # 存储log概率
        rewards =[] #
        for _ in range(steps_per_epoch):
            scores = policy_net(state)  # 作取动作的分数
            probabilities = torch.nn.functional.softmax(scores, dim = 1)  # 计算动作概率
            m = torch.distributions.Categorical(probabilities)
            #action = torch.argmax(probabilities).item()

            #action = m.sample()  # 米样一个山作

            # 使用 ε-greedy 选择动作
            if np.random.rand() < epsilon:
                action = torch.randint(0, probabilities.shape[1], (1,)).item()  # 10% 选择随机动作
            else:
                action = torch.argmax(probabilities).item()  # 90% 选择最优动作

            #print(f"Action: {action}")

            #print(action)
            # 根据灵样的动作更新状点
            step_size = 0.01
            new_state = state - step_size if action == 0 else state + step_size
            reward =  ((parabola(state) - parabola(new_state))*200)**3
            # print(new_state)
            # print(reward)
            #reward =  1/ if parabola(new_state)-1 >= 2 else reward == -1# 计算奖M
            #if parabola(new_state)-1 >= 2:
            #    reward = 1/(parabola(new_state)-1)
            #else:
            #    reward = -1

            #rewards.append(reward)
            rewards.append(torch.tensor([reward], dtype=torch.float32))  # 适配 PyTorch
            log_probs.append(m.log_prob(torch.tensor(action)))  # 计算 log 概率
            state = new_state  # 更新状态
            search_path.append(state.item())

        rewards = torch.cat(rewards)  # 拼接奖励
        log_probs = torch.cat(log_probs)  # 拼接概率  蒙特卡洛
        loss = -torch.sum(log_probs * rewards)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # 🔹 如果 loss 小于 min_loss，提前停止训练
        if abs(loss.item()) < min_loss:
            print(f"Early stopping at epoch {epoch + 1}, Loss: {loss.item()}")
            break  # 退出训练循环
        #
        #     log_prob = m.log_prob(action)  # i路LogM
        #     log_probs.append(log_prob)
        #     state = new_state  # 新状
        #     search_path.append(state.item())
        #
        # rewards = torch.cat(rewards) #拼接奖励
        # log_probs = torch.cat(log_probs) #拼接概率
        # loss = -torch.sum(torch.mul(log_probs,rewards))  # 计算损失
        # loss.backward()  # 汉向传
        # optimizer.step() # 火新参欢
        # **保存模型**
        #torch.save(policy_net.state_dict(), model_path)
        #print(f"Model saved to {model_path}")
        if (epoch + 1)% 100 == 0:
            print(f"Epoch {epoch +1}/{epochs}, Loss: {loss.item()}")

    # **保存模型**
    torch.save(policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return search_path

#train_PolicyNet(100, 100)


def test_policy(policy_net, steps=100):
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



import matplotlib.pyplot as plt

def plot_search_path(search_path):
    plt.figure(figsize=(8, 5))
    plt.plot(search_path, marker="o", linestyle="-", color="b", alpha=0.7, label="Search Path")
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.title("Policy Search Path")
    plt.legend()
    plt.grid(True)
    plt.show()


# 训练模型
train_PolicyNet(epochs=10000, steps_per_epoch=100, epsilon=0.1, min_loss=0.0001, model_path="C:/Users/28123/Desktop/policy_net.pth")

# 测试训练好的模型
#search_path = test_policy(policy_net, steps=1000)

# 可视化搜索路径
#plot_search_path(search_path)

#
# def load_model(model_path="policy_net.pth"):
#     model = PolicyNet()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # 设置为评估模式
#     print(f"Model loaded from {model_path}")
#     return model
#
# trained_model = load_model("C:/Users/28123/Desktop/policy_net.pth")
# search_path = test_policy(trained_model, steps=100)
# plot_search_path(search_path)
