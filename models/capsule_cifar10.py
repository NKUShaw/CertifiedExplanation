import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x, dim=-1):
    suqared_norm = (x ** 2 ).sum(dim=dim, keepdim=True)
    scale = suqared_norm / (1 + suqared_norm)
    return scale * (x / suqared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size, stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):
    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing, device):
        """
        Initialize the layer.
        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        temp_u_hat = u_hat.detach()
        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)
        for route_iter in range(self.num_routing - 1):
            c = b.softmax(dim=1)
            s = (c * temp_u_hat).sum(dim=2)
            v = squash(s)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)

        return v

class CapsNet(nn.Module):
    def __init__(self, device):
        super(CapsNet, self).__init__()
        self.device = device
        self.conv = nn.Conv2d(1, 256, 9)
        self.relu = nn.ReLU(inplace=True)
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=32 * 8 * 8,
                                    num_caps=10,
                                    dim_caps=16,
                                    num_routing=3,
                                    device=device)
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(10).to(self.device).index_select(dim=0, index=torch.argmax(logits, dim=1))
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))
        return logits, reconstruction