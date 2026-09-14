import torch
from torch.autograd import Function

class _GRL(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None

def grad_reverse(x, strength=1.0):
    return _GRL.apply(x, strength)

# Di dalam model:
# domain_logits = self.domain_head(grad_reverse(z, strength=1.0))

# Total loss
# loss = cls_loss + 0.10 * transform_loss + 0.05 * domain_loss