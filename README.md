# DQN CartPole Demo

使用 PyTorch 实现深度 Q 网络（DQN），解决 OpenAI Gymnasium 的 CartPole-v1 平衡控制问题。

## 环境
- Python 3.11
- PyTorch
- Gymnasium
- NumPy

## 运行
```bash
python dqn.py
```

## 算法要点
- 经验回放（Experience Replay）
- 目标网络（Target Network）
- ε-贪心策略探索

## 结果
- 在约 300 个 episode 内达到 475+ 的平均奖励，解决 CartPole 问题。
