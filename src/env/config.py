ENV_CONFIG = {
    "red_uav_num": 15,
    "red_usv_num": 5,
    "blue_uav_num": 12,
    "blue_usv_num": 3,
    "world_bounds": {
        "x": [-50, 250],
        "y": [-50, 250],
        "z": [0, 50],
    },
    "max_steps": 200,
    "red_base_reward": 180.0,
    "blue_base_penalty": -150.0,
    "out_of_bounds_penalty": -0.05,
    "agent_specs": {
        "uav": {
            "hp": 1,
            "firepower": 1,
            "speed": 10,
            "attack_range": 2,
            "detect_range": 20,
            "z": ">0"
        },
        "usv": {
            "hp": 3,
            "firepower": 3,
            "speed": 5,
            "attack_range": 80,
            "detect_range": 130,
            "z": "=0"
        },
        "base": {
            "hp": 1
        }
    },
    "record_trajectory": True,  # 是否记录轨迹
    "trajectory_output_dir": "trajectory_records",  # 轨迹输出目录
    # GPU优化相关配置
    "gpu_config": {
        "batch_size": 512,  # 针对4060ti优化
        "num_workers": 6,   # 数据加载并行度
        "pin_memory": True, # 加速GPU数据传输
        "mixed_precision": False,  # 是否启用混合精度（可选）
    }
}