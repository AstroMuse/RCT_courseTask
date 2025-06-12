import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ppo.obsDataProcess import ObsEncoder
# import time


class MLPPolicy(nn.Module):
    def __init__(self, embed_dim, act_dim, num_targets):
        super().__init__()
        self.encoder = ObsEncoder(embed_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.attack_logits = nn.Linear(128, num_targets + 1)

    def forward(self, entity_obs, valid_targets_mask=None):
        x = self.encoder(entity_obs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        attack_logits = self.attack_logits(x)
        # 确保攻击目标在有效范围内
        if valid_targets_mask is not None:
            # 将非法目标的logits设为一个很小的值
            invalid_mask = ~valid_targets_mask  # 取反获得非法目标掩码
            attack_logits = attack_logits.masked_fill(invalid_mask, -1e9)
        return mean, std, attack_logits


class ValueNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = ObsEncoder(embed_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, entity_obs):
        x = self.encoder(entity_obs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class AgentPolicy:
    def __init__(self, obs_dim, act_dim, num_targets, device):
        self.policy_net = MLPPolicy(obs_dim, act_dim, num_targets).to(device)
        self.value_net = ValueNet(embed_dim=obs_dim).to(device)
        self.device = device

        if device.type == 'cuda':
            self.policy_net = self.policy_net.half().float()  # 预热混合精度
            self.value_net = self.value_net.half().float()

    def act(self, obs, valid_targets_mask=None):
        # 检查obs是否为列表，如果不是，将其视为单个观测
        is_single_obs = not isinstance(obs, list)
        if is_single_obs:
            obs = [obs]
        if valid_targets_mask is not None:
            valid_targets_mask = torch.tensor(valid_targets_mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():  # 推理时不需要梯度
            mean, std, attack_logits = self.policy_net(obs, valid_targets_mask)

            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            attack_dist = torch.distributions.Categorical(logits=attack_logits)
            attack_target = attack_dist.sample()
            attack_log_prob = attack_dist.log_prob(attack_target)

        # 如果是单个观测，返回单个结果
        if is_single_obs:
            return (action[0].detach().cpu().numpy(), attack_target[0].item()), (
                    log_prob[0] + attack_log_prob[0]).detach(), dist.entropy().sum(dim=-1)[0].detach()
        else:
            return (action.detach().cpu().numpy(), attack_target.item()), (
                    log_prob + attack_log_prob).detach(), dist.entropy().sum(dim=-1).detach()

    def evaluate(self, obs, act, attack_target, valid_targets_mask=None):
        # 不需要转换 obs 为张量，因为 ObsEncoder 的 forward 方法已经处理了字典结构
        # 移除对 obs 形状的检查和调整，因为新的观测结构不再是张量
        # act = torch.tensor(act, dtype=torch.float32).to(self.device)
        # attack_target = torch.tensor(attack_target, dtype=torch.long).to(self.device)

        if torch.is_tensor(act):
            act = act.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            act = torch.tensor(act, dtype=torch.float32).to(self.device)

        if torch.is_tensor(attack_target):
            attack_target = attack_target.detach().clone().to(dtype=torch.long, device=self.device)
        else:
            attack_target = torch.tensor(attack_target, dtype=torch.long).to(self.device)

        if valid_targets_mask is not None:
            valid_targets_mask = torch.tensor(valid_targets_mask, dtype=torch.bool).to(self.device)

        mean, std, attack_logits = self.policy_net(obs, valid_targets_mask)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(act).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        attack_dist = torch.distributions.Categorical(logits=attack_logits)
        attack_log_prob = attack_dist.log_prob(attack_target)
        attack_entropy = attack_dist.entropy()
        total_log_prob = log_prob + attack_log_prob
        total_entropy = entropy + attack_entropy

        value = self.value_net(obs).squeeze(-1)

        return total_log_prob, total_entropy, value


# 初始化四类策略网络接口
class PolicyManager:
    def __init__(self, obs_dim, act_dim, red_targets_num, blue_targets_num, device):
        self.red_uav = AgentPolicy(obs_dim, act_dim, red_targets_num, device)
        self.red_usv = AgentPolicy(obs_dim, act_dim, red_targets_num, device)
        self.blue_uav = AgentPolicy(obs_dim, act_dim, blue_targets_num, device)
        self.blue_usv = AgentPolicy(obs_dim, act_dim, blue_targets_num, device)

    def get_policy(self, team, agent_type):
        if team == 'red' and agent_type == 'uav':
            return self.red_uav
        elif team == 'red' and agent_type == 'usv':
            return self.red_usv
        elif team == 'blue' and agent_type == 'uav':
            return self.blue_uav
        elif team == 'blue' and agent_type == 'usv':
            return self.blue_usv
        else:
            raise ValueError("Unknown team or agent_type")
