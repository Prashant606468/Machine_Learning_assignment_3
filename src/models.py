"""All architectures. Parameter counts are printed by `python src/models.py`."""
import torch
import torch.nn as nn


def _mlp_layers(d_in, hidden, n_out=10):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Flatten 3x32x32 -> hidden layers (Linear-BN-ReLU) -> 10 logits. No spatial prior:
    the function class is invariant to a fixed permutation of input pixels."""
    def __init__(self, hidden):
        super().__init__()
        self.net = _mlp_layers(3 * 32 * 32, hidden)

    def forward(self, x):
        return self.net(x.flatten(1))


class LocalMLP(nn.Module):
    """MLP with ONE added spatial prior: a 48->token_dim linear layer shared across all
    non-overlapping 4x4 patches (equivalent to conv 4x4, stride 4). With token_dim=48 the
    flattened size is 64*48=3072, so the head is exactly MLP-base and the model has only
    2,352+BN extra parameters. On pixel-permuted inputs the prior becomes meaningless."""
    def __init__(self, patch=4, token_dim=48, hidden=(512, 512)):
        super().__init__()
        self.patch = nn.Conv2d(3, token_dim, kernel_size=patch, stride=patch)
        self.bn = nn.BatchNorm2d(token_dim)
        n_tok = (32 // patch) ** 2
        self.head = _mlp_layers(n_tok * token_dim, list(hidden))

    def forward(self, x):
        return self.head(torch.relu(self.bn(self.patch(x))).flatten(1))


class CNN(nn.Module):
    """VGG-style: 3 stages of two 3x3 conv-BN-ReLU, 2x2 max-pool after each stage,
    global average pool, linear classifier. Widths chosen to match MLP-base params."""
    def __init__(self, widths=(64, 128, 352)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            for _ in range(2):
                layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
                c = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(c, 10)

    def embed(self, x):
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.embed(x))


class ConvAE(nn.Module):
    """Convolutional autoencoder. Only the bottleneck width `d` varies between models.
    enc: 3x32x32 -> 32x16x16 -> 64x8x8 -> 128x4x4 -> Linear(2048, d)
    dec: Linear(d, 2048) -> 128x4x4 -> 64x8x8 -> 32x16x16 -> 3x32x32 (sigmoid)."""
    def __init__(self, d):
        super().__init__()
        def down(i, o):
            return [nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.enc_conv = nn.Sequential(*down(3, 32), *down(32, 64), *down(64, 128))
        self.enc_fc = nn.Linear(128 * 16, d)
        self.dec_fc = nn.Sequential(nn.Linear(d, 128 * 16), nn.ReLU(inplace=True))
        self.dec_conv = nn.Sequential(*up(128, 64), *up(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
        self.d = d

    def encode(self, x):
        return self.enc_fc(self.enc_conv(x).flatten(1))

    def decode(self, z):
        return self.dec_conv(self.dec_fc(z).view(-1, 128, 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    from common import count_params, count_macs, load_cfg
    cfg = load_cfg()
    q1, q2 = cfg["q1"], cfg["q2"]
    ms = {"MLP-base": MLP(q1["mlp_base_hidden"]), "MLP-wide": MLP(q1["mlp_wide_hidden"]),
          "LocalMLP": LocalMLP(q1["local_patch"], q1["local_token_dim"], q1["local_hidden"]),
          "CNN": CNN(q2["cnn_widths"])}
    ms.update({f"AE-{d}": ConvAE(d) for d in cfg["q3"]["bottlenecks"]})
    for k, m in ms.items():
        print(f"{k:10s} params {count_params(m):>10,d}  MACs {count_macs(m):>12,d}")
