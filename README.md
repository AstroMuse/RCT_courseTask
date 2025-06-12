# UAV-USV Multi-Agent Combat Simulation

<div align="center">


**基于深度强化学习的无人机-无人船多智能体协同作战仿真系统**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/username/UAV_USV_RL_V6.svg)](https://github.com/username/UAV_USV_RL_V6/stargazers)

[English](README_EN.md) | 简体中文

</div>

## 🎯 项目简介

本项目是一个基于**深度强化学习**的多智能体协同作战仿真系统，模拟红蓝双方无人机(UAV)和无人船(USV)的对抗场景。系统采用**PPO(Proximal Policy Optimization)**算法训练智能体，实现端到端的路径规划、目标分配和火力控制。

### ✨ 核心特性

- 🚁 **多智能体协同**: 支持UAV和USV混合编队作战
- 🧠 **端到端学习**: 集成路径规划、目标分配、火力分配的统一决策
- 🌍 **3D战场环境**: 完整的三维作战空间模拟
- 📊 **实时可视化**: 基于Web的3D战场态势显示
- 📈 **轨迹记录**: 完整的训练和测试轨迹数据记录
- ⚡ **GPU加速**: 针对NVIDIA GPU优化的训练流程
- 🎮 **交互式回放**: 支持暂停、快进、倒退的战斗回放

## 🏗️ 项目架构

```
UAV_USV_RL_V6/
├── 📁 src/                          # 核心源代码
│   ├── 📁 agents/                   # 智能体定义
│   │   └── 📄 agent.py             # Agent和Base类实现
│   ├── 📁 env/                      # 环境模块
│   │   ├── 📄 config.py            # 环境配置参数
│   │   └── 📄 world.py             # 战场环境实现
│   ├── 📁 ppo/                      # PPO算法实现
│   │   ├── 📄 model.py             # 神经网络模型
│   │   ├── 📄 ppo.py               # PPO训练算法
│   │   └── 📄 obsDataProcess.py    # 观测数据处理
│   └── 📁 utils/                    # 工具模块
│       ├── 📄 trajectory_manager.py # 轨迹记录管理
│       └── 📄 plot_kl_rew.py       # 训练曲线绘制
├── 📁 scripts/                      # 脚本文件
│   ├── 📄 train.py                 # 训练脚本
│   ├── 📄 visualize.html           # 3D可视化界面
│   ├── 📄 printDebug.py            # 调试工具
│   ├── 📁 weights/                 # 模型权重存储
│   ├── 📁 log/                     # 训练日志
│   └── 📁 trajectory_records/      # 轨迹记录文件
├── 📄 README.md                     # 项目说明文档
└── 📄 README.png                    # 项目横幅图片
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.9+
- **PyTorch**: 2.7.0
- **CUDA**: 支持CUDA的GPU (推荐)
- **内存**: 8GB+ RAM
- **显存**: 4GB+ VRAM (推荐RTX 4060Ti或更高)

### 安装依赖

```bash
# 安装PyTorch (CUDA版本)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install numpy matplotlib typing
```

### 训练模型

```bash
cd scripts
python train.py
```

### 可视化结果

1. 在浏览器中打开 `scripts/visualize.html`
2. 点击"选择文件"加载轨迹记录文件
3. 使用控制面板观看3D战场回放

## ⚙️ 配置说明

### 智能体配置

| 类型 | 数量 | HP | 火力 | 速度 | 攻击范围 | 探测范围 | 高度约束 |
|------|------|----|----|------|----------|----------|----------|
| 红方UAV | 15 | 1 | 1 | 10 | 2 | 20 | z > 0 |
| 红方USV | 5 | 3 | 3 | 5 | 80 | 130 | z = 0 |
| 蓝方UAV | 12 | 1 | 1 | 10 | 2 | 20 | z > 0 |
| 蓝方USV | 3 | 3 | 3 | 5 | 80 | 130 | z = 0 |

### 战场环境

- **世界边界**: X[-50, 250], Y[-50, 250], Z[0, 50]
- **最大步数**: 200步
- **红方出生点**: [0, 0, z] (UAV: z=1, USV: z=0)
- **蓝方出生点**: [200, 200, z] (UAV: z=1, USV: z=0)
- **蓝方基地**: 位置[201, 201, 0], HP=1

### 奖励机制

#### 基础奖励
- **红方摧毁蓝方基地**: +180.0
- **蓝方防守成功**: -150.0 (红方未能摧毁基地时的惩罚)
- **越界惩罚**: -0.05 (智能体超出战场边界)
- **攻击成功奖励**: +0.5 (成功摧毁敌方智能体)

#### 距离奖励机制
- **红方距离奖励**: 最大+0.3
  - 距离蓝方基地30米内：获得最大奖励0.3
  - 在正确方向前进：根据进度线性递减(0.3 × 进度比例)
  - 越过基地或背离方向：无奖励
  - 采用方向感知算法，防止智能体"穿越"基地行为

#### 奖励设计特点
- **方向导向**: 红方距离奖励基于投影算法，确保智能体朝向目标前进
- **防穿越机制**: 智能体越过基地后不再获得距离奖励
- **平衡性**: 攻击奖励与防守惩罚比例合理，促进积极对抗
- **边界约束**: 越界惩罚较小(-0.05)，避免过度限制探索

## 🧠 算法特点

### PPO强化学习

- **策略网络**: MLP架构，支持连续动作空间
- **价值网络**: 独立的价值函数估计
- **观测编码**: 基于Transformer的多实体观测处理
- **目标分配**: 离散动作空间的攻击目标选择
- **GAE优势估计**: 广义优势估计提升训练稳定性

### 多智能体协调

- **分布式训练**: 每个智能体独立决策
- **局部观测**: 基于探测范围的部分可观测环境
- **实体编码**: 区分自身、友军、敌军的观测编码
- **掩码机制**: 确保攻击目标在有效范围内

### GPU优化

- **批处理优化**: 针对RTX 4060Ti优化的批处理大小(512)
- **内存管理**: 高效的数据加载和传输
- **混合精度**: 可选的FP16训练支持
- **CUDNN优化**: 启用CUDNN基准测试模式

## 📊 训练监控

系统提供完整的训练监控功能：

- **📈 实时日志**: 详细的训练过程记录
- **📉 奖励曲线**: 训练奖励和KL散度可视化
- **🎯 轨迹记录**: 每个episode的完整轨迹数据
- **💾 模型保存**: 自动保存最佳模型权重
- **🔍 调试工具**: 详细的episode统计信息

### 日志文件

```bash
log/
├── train_log_20241212_220000.log  # 训练日志
└── ...
```

### 权重文件

```bash
weights/
├── red_policy_best.pth           # 红方最佳策略
├── blue_policy_best.pth          # 蓝方最佳策略
└── ...
```

## 🎮 可视化功能

### 3D战场显示

- **🎨 实体渲染**: 不同颜色区分红蓝双方
  - 🔴 红方UAV: 红色球体
  - 🔵 蓝方UAV: 蓝色球体
  - 🟥 红方USV: 红色立方体
  - 🟦 蓝方USV: 蓝色立方体
  - ⚫ 蓝方基地: 黑色立方体
- **📍 轨迹追踪**: 智能体移动路径显示
- **📊 状态信息**: 实时HP、存活状态
- **🎛️ 回放控制**: 支持暂停、快进、倒退、速度调节

### 控制面板

- **文件加载**: 支持JSON格式的轨迹文件
- **播放控制**: 播放/暂停、步进、重置
- **速度调节**: 1x - 10x播放速度
- **视角控制**: 鼠标拖拽旋转、滚轮缩放
- **信息显示**: 当前步数、智能体状态

## 🔧 自定义配置

### 修改智能体数量

在 `src/env/config.py` 中修改：

```python
ENV_CONFIG = {
    "red_uav_num": 15,    # 红方UAV数量
    "red_usv_num": 5,     # 红方USV数量
    "blue_uav_num": 12,   # 蓝方UAV数量
    "blue_usv_num": 3,    # 蓝方USV数量
    # ...
}
```

### 调整奖励函数

在 `src/env/world.py` 中修改奖励计算逻辑：

```python
# 距离奖励
if distance_to_base <= 30:
    distance_reward = 0.3
# 攻击奖励
for agent_id in successful_red_attackers:
    rewards[agent_id] += 0.5
```

### GPU配置优化

在 `src/env/config.py` 中调整GPU设置：

```python
"gpu_config": {
    "batch_size": 512,        # 根据显存调整
    "num_workers": 6,         # CPU核心数
    "pin_memory": True,       # 加速数据传输
    "mixed_precision": False, # 混合精度训练
}
```

## 📈 性能基准

### 训练性能

| 硬件配置 | 训练速度 | 内存使用 | 显存使用 |
|----------|----------|----------|----------|
| RTX 4060Ti | ~100 steps/s | 6GB | 3.5GB |
| RTX 3080 | ~150 steps/s | 8GB | 5GB |
| CPU Only | ~20 steps/s | 4GB | - |

### 收敛性能

- **收敛步数**: ~50,000 steps
- **最佳胜率**: 红方 65% vs 蓝方 35%
- **平均episode长度**: 150-180 steps

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **Fork** 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 **Pull Request**

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🎨 代码优化
- 🧪 测试用例
- 🌐 国际化支持

### 开发规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档
- 确保代码通过现有测试
- 新功能需要添加相应测试

## 📄 许可证

本项目采用 **MIT许可证** - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) 团队提供的深度学习框架
- [OpenAI](https://openai.com/) 的PPO算法实现参考
- 多智能体强化学习社区的宝贵建议
- [Three.js](https://threejs.org/) 提供的3D可视化支持

## 📞 联系我们

- **讨论交流**: [Discussions](https://pd.qq.com/s/3dnwpgvu4?b=9)
- **邮箱**: Gfmiao@proton.me

## 📊 项目统计

![GitHub stars](https://img.shields.io/github/stars/username/UAV_USV_RL_V6.svg?style=social)
![GitHub forks](https://img.shields.io/github/forks/username/UAV_USV_RL_V6.svg?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/username/UAV_USV_RL_V6.svg?style=social)

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 ⭐ Star！**

[⬆ 回到顶部](#uav-usv-multi-agent-combat-simulation)

</div>
        
