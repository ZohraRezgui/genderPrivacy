import torch
from torch import nn

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class P1Loss(nn.Module):
    def __init__(self):
        super(P1Loss, self).__init__()

    def forward(self, proj_features, female_idx, male_idx):
        female_features = proj_features[female_idx, :]
        male_features = proj_features[male_idx, :]
        mean_female_features = l2_norm(female_features.mean(dim=0), axis=1)
        mean_male_features = l2_norm(male_features.mean(dim=0), axis=1)

        loss_p1 = (mean_female_features * mean_male_features).sum(dim=-1, keepdim=True).clamp(-1, 1).acos()
        return loss_p1
    
class P2Loss(nn.Module):
    def __init__(self, label_f, label_m):
        super(P2Loss, self).__init__()
        self.label_f = label_f
        self.label_m = label_m
    
    def forward(self, header, proj_features, female_idx, male_idx):
        female_features = proj_features[female_idx, :]
        male_features = proj_features[male_idx, :]
        norm_kernel = l2_norm(header.kernel, axis=0)
        W_F = norm_kernel[:, self.label_f].mean(dim=1)
        W_M = norm_kernel[:, self.label_m].mean(dim=1)
        W_F_norm = l2_norm(W_F, axis=0).unsqueeze(1)
        W_M_norm = l2_norm(W_M, axis=0).unsqueeze(1)

        female_featuresxmale_weights = torch.mm(female_features.squeeze(1), W_M_norm).clamp(-1, 1).acos()
        male_featuresxfemale_weights = torch.mm(male_features.squeeze(1), W_F_norm).clamp(-1, 1).acos()

        loss_p2 = female_featuresxmale_weights.mean() + male_featuresxfemale_weights.mean()

        return loss_p2

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta





