from torch import nn
import torch


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False) 

    def forward(self, img):
        imgs = img.chunk(img.shape[1], dim=1)
        edge_imgs = []
        for channel in imgs:
            x = self.filter(channel)
            x = torch.mul(x, x)
            x = torch.sum(x, dim=1, keepdim=True)
            x = torch.sqrt(x)
            edge_imgs.append(x)
        edge_img = torch.cat(edge_imgs, dim=1)
        return edge_img
