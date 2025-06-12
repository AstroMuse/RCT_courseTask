import torch
import torch.nn as nn
# import time


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ObsEncoder(nn.Module):
    def __init__(self, embed_dim=0, num_heads=16, num_layers=2):
        super().__init__()
        self.type_embedding = nn.Embedding(3, 4)
        self.self_encoder = MLP(3 + 2 + 4, embed_dim)  # position + hp/firepower + type_emb
        self.friend_encoder = MLP(3 + 2 + 4, embed_dim)
        self.enemy_encoder = MLP(3 + 4, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, obs_batch):
        device = next(self.parameters()).device
        if not isinstance(obs_batch, list):
            obs_batch = [obs_batch]

        # batch_size = len(obs_batch)
        # print(f"batch_size:{batch_size}")
        all_tokens = []
        all_masks = []

        for obs in obs_batch:
            tokens = []
            masks = []

            # 自身信息
            self_type = 0 if obs["self"]["type"] == "uav" else (1 if obs["self"]["type"] == "usv" else 2)
            type_emb = self.type_embedding(torch.tensor(self_type, device=device))
            self_feat = torch.cat([
                torch.tensor(obs["self"]["position"], dtype=torch.float32, device=device),
                torch.tensor([obs["self"]["hp"], obs["self"]["firepower"]], dtype=torch.float32, device=device),
                type_emb
            ])
            tokens.append(self.self_encoder(self_feat))
            masks.append(False)

            # 友方信息
            for friend in obs["friends"]:
                friend_type = 0 if friend["type"] == "uav" else (1 if friend["type"] == "usv" else 2)
                type_emb = self.type_embedding(torch.tensor(friend_type, device=device))
                feat = torch.cat([
                    torch.tensor(friend["position"], dtype=torch.float32, device=device),
                    torch.tensor([friend["hp"], friend["firepower"]], dtype=torch.float32, device=device),
                    type_emb
                ])
                tokens.append(self.friend_encoder(feat))
                masks.append(False)

            # 敌方信息
            for enemy in obs["enemies"]:
                enemy_type = 0 if enemy["type"] == "uav" else (1 if enemy["type"] == "usv" else 2)
                type_emb = self.type_embedding(torch.tensor(enemy_type, device=device))
                feat = torch.cat([
                    torch.tensor(enemy["position"], dtype=torch.float32, device=device),
                    type_emb
                ])
                tokens.append(self.enemy_encoder(feat))
                masks.append(False)

            all_tokens.append(torch.stack(tokens))  # (L_i, D)
            all_masks.append(torch.tensor(masks, dtype=torch.bool, device=device))  # (L_i,)

        # Padding 到统一长度
        max_len = max(t.shape[0] for t in all_tokens)
        padded_tokens = []
        padded_masks = []

        for tokens, masks in zip(all_tokens, all_masks):
            pad_len = max_len - tokens.shape[0]
            if pad_len > 0:
                pad_tokens = torch.zeros(pad_len, tokens.shape[1], device=device)
                pad_mask = torch.ones(pad_len, dtype=torch.bool, device=device)
                tokens = torch.cat([tokens, pad_tokens], dim=0)
                masks = torch.cat([masks, pad_mask], dim=0)
            padded_tokens.append(tokens)
            padded_masks.append(masks)

        tokens_batch = torch.stack(padded_tokens)  # (B, L, D)
        masks_batch = torch.stack(padded_masks)  # (B, L)

        # Transformer 编码
        encoded = self.transformer_encoder(tokens_batch, src_key_padding_mask=masks_batch)

        # 全局平均池化（忽略 padding）
        masks_expanded = masks_batch.unsqueeze(-1).expand_as(encoded)  # (B, L, D)
        encoded_masked = encoded.masked_fill(masks_expanded, 0.0)
        valid_counts = (~masks_batch).sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        global_repr = encoded_masked.sum(dim=1) / valid_counts  # (B, D)

        return global_repr
