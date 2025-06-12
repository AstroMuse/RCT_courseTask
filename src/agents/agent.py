import numpy as np


class Agent:
    def __init__(self, id, team, type, start_pos, config):
        self.id = id
        self.team = team  # 'red' or 'blue'
        self.type = type  # 'uav' or 'usv'
        self.config = config
        self.position = np.array(start_pos, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.hp = config["hp"]
        self.firepower = config["firepower"]
        self.speed = config["speed"]
        self.detect_range = config["detect_range"]
        self.attack_range = config["attack_range"]
        self.alive = True
        self.destroyed = False

    def step(self, action):
        if not self.alive:
            return
        # 连续动作: action = [dx, dy, dz]
        direction = np.clip(action, -1, 1)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        move = direction * self.speed
        self.position += move

        # UAV 约束 z > 0，USV 约束 z == 0
        if self.type == 'uav':
            self.position[2] = max(self.position[2], 0.01)
        elif self.type == 'usv':
            self.position[2] = 0.0

    def distance_to(self, other):
        return np.linalg.norm(self.position - other.position)

    def in_attack_range(self, other):
        return self.distance_to(other) <= self.attack_range

    def can_detect(self, other):
        return self.distance_to(other) <= self.detect_range

    def receive_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False
            self.destroyed = True  # 明确标记为已销毁


class Base:
    def __init__(self, position, hp):
        self.position = np.array(position, dtype=np.float32)
        self.hp = hp
        self.alive = True

    def receive_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False
