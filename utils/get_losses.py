import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLoss(nn.Module):
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-3
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)
        self.margin_loss = margin_loss
        self.reconstruction_loss = reconstruction_loss
        # Orthogonal loss : Cosine similarity between filters / Or maximize L2 distance between filters
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        #  features are normalized
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]  # extend dim
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())
        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs
        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        return loss