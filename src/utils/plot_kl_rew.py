import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(rewards, kl_divs, episode, final=False):
    """原有的简单绘制函数，保持兼容性"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制奖励曲线
    ax1.plot(rewards, 'b-', alpha=0.7, label='Episode Reward')
    if len(rewards) > 10:
        # 添加移动平均线
        window = min(50, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制KL散度曲线
    ax2.plot(kl_divs, 'g-', alpha=0.7, label='KL Divergence')
    if len(kl_divs) > 10:
        window = min(50, len(kl_divs) // 10)
        moving_avg_kl = np.convolve(kl_divs, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(kl_divs)), moving_avg_kl, 'orange', linewidth=2,
                 label=f'Moving Average ({window})')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Divergence Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if final:
        plt.savefig(f'training_curves_final.png', dpi=300, bbox_inches='tight')
        print("最终训练曲线已保存为 training_curves_final.png")
    else:
        plt.savefig(f'training_curves_ep{episode}.png', dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存为 training_curves_ep{episode}.png")

    plt.close()  # 关闭图形以释放内存


def plot_layered_training_curves(reward_history, win_statistics, kl_divs, episode, final=False):
    """分层可视化训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 第一层：团队对比视图

    # (0,0) 团队奖励对比曲线
    axes[0, 0].plot(reward_history['red_team'], 'r-', linewidth=2, label='Red Team', alpha=0.8)
    axes[0, 0].plot(reward_history['blue_team'], 'b-', linewidth=2, label='Blue Team', alpha=0.8)

    # 添加移动平均线
    if len(reward_history['red_team']) > 10:
        window = min(20, len(reward_history['red_team']) // 5)
        red_ma = np.convolve(reward_history['red_team'], np.ones(window) / window, mode='valid')
        blue_ma = np.convolve(reward_history['blue_team'], np.ones(window) / window, mode='valid')
        axes[0, 0].plot(range(window - 1, len(reward_history['red_team'])), red_ma, 'r--', alpha=0.6,
                        label=f'Red MA({window})')
        axes[0, 0].plot(range(window - 1, len(reward_history['blue_team'])), blue_ma, 'b--', alpha=0.6,
                        label=f'Blue MA({window})')

    axes[0, 0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Team Reward')
    axes[0, 0].set_title('Team Reward Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) 胜率统计柱状图
    win_rates = [win_statistics['red_win_rate'][-1] if win_statistics['red_win_rate'] else 0,
                 win_statistics['blue_win_rate'][-1] if win_statistics['blue_win_rate'] else 0]
    bars = axes[0, 1].bar(['Red Team', 'Blue Team'], win_rates, color=['red', 'blue'], alpha=0.7)
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].set_title(f'Win Rate Statistics (Episode {episode + 1})')
    axes[0, 1].set_ylim(0, 1)

    # 在柱状图上添加数值标签
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')

    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # (0,2) 胜率演化曲线
    if win_statistics['red_win_rate'] and win_statistics['blue_win_rate']:
        axes[0, 2].plot(win_statistics['red_win_rate'], 'r-', linewidth=2, label='Red Win Rate')
        axes[0, 2].plot(win_statistics['blue_win_rate'], 'b-', linewidth=2, label='Blue Win Rate')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].set_title('Win Rate Evolution')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 第二层：智能体类型分析

    # (1,0) 异构智能体性能对比 - UAV vs USV
    axes[1, 0].plot(reward_history['red_uav'], 'r-', linewidth=2, label='Red UAV', alpha=0.8)
    axes[1, 0].plot(reward_history['red_usv'], 'r:', linewidth=2, label='Red USV', alpha=0.8)
    axes[1, 0].plot(reward_history['blue_uav'], 'b-', linewidth=2, label='Blue UAV', alpha=0.8)
    axes[1, 0].plot(reward_history['blue_usv'], 'b:', linewidth=2, label='Blue USV', alpha=0.8)
    axes[1, 0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].set_title('Heterogeneous Agent Performance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1) 红蓝双方UAV对比
    axes[1, 1].plot(reward_history['red_uav'], 'r-', linewidth=2, label='Red UAV')
    axes[1, 1].plot(reward_history['blue_uav'], 'b-', linewidth=2, label='Blue UAV')
    axes[1, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].set_title('UAV Performance Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # (1,2) 红蓝双方USV对比
    axes[1, 2].plot(reward_history['red_usv'], 'r:', linewidth=2, label='Red USV')
    axes[1, 2].plot(reward_history['blue_usv'], 'b:', linewidth=2, label='Blue USV')
    axes[1, 2].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Average Reward')
    axes[1, 2].set_title('USV Performance Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if final:
        plt.savefig('layered_training_curves_final.png', dpi=300, bbox_inches='tight')
        print("最终分层训练曲线已保存为 layered_training_curves_final.png")
    else:
        plt.savefig(f'layered_training_curves_ep{episode}.png', dpi=300, bbox_inches='tight')
        print(f"分层训练曲线已保存为 layered_training_curves_ep{episode}.png")

    plt.close()  # 关闭图形以释放内存


def plot_kl_divergence_curve(kl_divs, episode, final=False):
    """单独绘制KL散度曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(kl_divs, 'g-', alpha=0.7, label='KL Divergence')
    if len(kl_divs) > 10:
        window = min(50, len(kl_divs) // 10)
        moving_avg_kl = np.convolve(kl_divs, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(kl_divs)), moving_avg_kl, 'orange', linewidth=2,
                label=f'Moving Average ({window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if final:
        plt.savefig('kl_divergence_final.png', dpi=300, bbox_inches='tight')
        print("最终KL散度曲线已保存为 kl_divergence_final.png")
    else:
        plt.savefig(f'kl_divergence_ep{episode}.png', dpi=300, bbox_inches='tight')
        print(f"KL散度曲线已保存为 kl_divergence_ep{episode}.png")

    plt.close()