from torch import nn
import torch.nn.functional as F
import torch

class Bce_Loss(nn.Module):
    def __init__(self):
        super(Bce_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.binary_cross_entropy(x, label)
        return loss

class Weighed_Bce_Loss(nn.Module):
    def __init__(self):
        super(Weighed_Bce_Loss, self).__init__()

    def forward(self, x, label):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        label = label.view(-1, 1, label.shape[1], label.shape[2])
        label_t = (label == 1).float()
        label_f = (label == 0).float()
        p = torch.sum(label_t) / (torch.sum(label_t) + torch.sum(label_f))
        w = torch.zeros_like(label)
        w[label == 1] = p
        w[label == 0] = 1 - p
        loss = F.binary_cross_entropy(x, label, weight=w)
        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, thres, min_kept=100000):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)

    def forward(self, x, label):
        pixel_losses = F.binary_cross_entropy(x, label, reduction='none').contiguous().view(-1)
        pred, ind = x.contiguous().view(-1).sort()
        pixel_losses = pixel_losses[pred < self.thresh]
        return pixel_losses.mean()

class Cls_Loss(nn.Module):
    def __init__(self):
        super(Cls_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.binary_cross_entropy(x, label)
        return loss

class S_Loss(nn.Module):
    def __init__(self):
        super(S_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.smooth_l1_loss(x, label)
        return loss
        
class adj_Loss(nn.Module):
    def __init__(self):
        super(adj_Loss, self).__init__()

    def forward(self, x, label):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        label = label.view(-1, 1, label.shape[1], label.shape[2])
        loss = F.binary_cross_entropy(x, label)
        return loss
#############################################
### /home/litengpeng/CODE/co-segmentation/MaCoSNet-pytorch-master/model/
class Loss2(nn.Module):
    def __init__(self):
        super(Loss2, self).__init__()
        self.loss_wbce = Weighed_Bce_Loss()
        self.loss_s = S_Loss()
        self.w_bce = 1
        self.w_smooth = 1

    def forward(self, x, label):
        m_loss = self.loss_wbce(x, label) * self.w_bce
        s_loss = self.loss_s(x, label) * self.w_smooth
        loss = m_loss + s_loss

        return loss, m_loss, s_loss

if __name__ == '__main__':
    x = torch.rand(4, 1, 224, 224)
    label = torch.zeros(4, 1, 224, 224)
    label[:, :, 30:80, 30:80] = 1
    loss = OhemCrossEntropy()
    l = loss(x, label)