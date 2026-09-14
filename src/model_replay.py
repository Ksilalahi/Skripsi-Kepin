import torch
import torch.nn as nn
from torchvision.models import resnet18

from grl import grad_reverse

class ReplayResNet(nn.Module):
    def __init__(self, n_transform=2, n_domain=2, embedding_dim=256):
        super().__init__()
        net = resnet18(weights=None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(nn.Linear(dim, embedding_dim),
                                  nn.LayerNorm(embedding_dim), nn.ReLU())
        self.class_head = nn.Linear(embedding_dim, 2)
        self.transform_head = nn.Linear(embedding_dim, n_transform)

        self.domain_head = nn.Sequential(
            nn.Linear(embedding_dim,128),
            nn.ReLU(),
            nn.Linear(128,n_domain)
        )

    def forward(self,x,grl_strength=1.0):
        backbone_feature = self.backbone(x)
        z = self.proj(backbone_feature)
        class_logits = self.class_head(z)
        transform_logits = self.transform_head(z)

        # Domain-adversarial branch
        z_reverse = grad_reverse(z,strength=grl_strength)

        domain_logits = self.domain_head(
            z_reverse)

        return (class_logits,transform_logits,domain_logits,z)