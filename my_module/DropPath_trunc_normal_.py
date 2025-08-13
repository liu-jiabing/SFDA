import torch
import torch.nn.functional as F

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.5):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + torch.rand_like(x)  # Uniform [0, 1)
            binary_mask = torch.floor(random_tensor)  # Random 0 or 1
            output = x / keep_prob * binary_mask
            return output
        return x
