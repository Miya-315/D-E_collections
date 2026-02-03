# 1. PPO
Policy model是根据rewards model - >weights的update
sft模型（reference model）作为初始化，但是weight是不断的update
![alt text](image-1.png)
## 1. Actor model/Policy model
1. 采样阶段： 我们用当前的模型$\pi_{old}$去跑出一堆经验。
2. 训练阶段： 我们更新模型，得到了更聪明的 $\pi_{new}$
问题：强化学习是 On-policy（即时策略） 算法，理论上你必须用 $\pi_{new}$产生的经验来训练 $\pi_{new}$。但这意味着你每更新一次参数，之前辛辛苦苦采样的数据就全作废了，必须重新采样。这在训练 7B 模型时简直是灾难，因为采样（推理）非常慢。
## 2. 