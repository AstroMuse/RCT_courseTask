import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ppo.model import AgentPolicy
import time


class PPOAgent:
    def __init__(self, policy: AgentPolicy, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, value_coef=0.5,
                 entropy_coef=0.01):
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.policy_optimizer = optim.Adam(policy.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(policy.value_net.parameters(), lr=lr)

        # GPU优化的优化器设置
        self.policy_optimizer = optim.Adam(policy.policy_net.parameters(), lr=lr, eps=1e-5)
        self.value_optimizer = optim.Adam(policy.value_net.parameters(), lr=lr, eps=1e-5)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, obs, acts, log_probs_old, returns, advantages, epochs=3, batch_size=1024):
        # 预先将所有数据转移到GPU
        device = self.policy.device

        # 拆分动作：acts 是 [(move_action, attack_target), ...]
        move_actions, attack_targets = zip(*acts)
        move_actions = torch.tensor(np.array(move_actions), dtype=torch.float32, device=device)
        attack_targets = torch.tensor(np.array(attack_targets), dtype=torch.long, device=device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

        # 标准化advantages以提升训练稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(obs)
        total_kl_div = 0.0
        num_updates = 0
        for epoch in range(epochs):
            print(f"[epoch {epoch}] update中...")
            indices = torch.randperm(dataset_size, device=device)  # 在GPU上生成随机索引
            for start in range(0, dataset_size, batch_size):
                print(f"[start {start}] update中...")
                print(f"[dataset_size: {dataset_size}] ")
                end = min(start + batch_size, dataset_size)  # 防止超出数据集大小
                mb_idx = indices[start:end]

                # mb_obs = [obs[i] for i in mb_idx]  # 保持字典结构
                mb_obs = [obs[i] for i in mb_idx.cpu().numpy()]  # 只在需要时转回CPU
                mb_move_actions = move_actions[mb_idx]
                mb_attack_targets = attack_targets[mb_idx]
                mb_log_probs_old = log_probs_old[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                log_probs, entropy, values = self.policy.evaluate(mb_obs, mb_move_actions, mb_attack_targets)
                # 计算KL散度
                kl_div = (mb_log_probs_old - log_probs).mean().item()
                total_kl_div += kl_div
                num_updates += 1

                # 其余部分保持不变
                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, mb_returns)
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                debug_start = time.time()
                total_loss.backward()
                debug_time = time.time() - debug_start
                print(f"[debug_time:{debug_time}]")

                torch.nn.utils.clip_grad_norm_(self.policy.policy_net.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.policy.value_net.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                # GPU内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_kl_div = total_kl_div / num_updates if num_updates > 0 else 0.0
        return avg_kl_div
