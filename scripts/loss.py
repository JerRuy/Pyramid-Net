
import torch
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a





class diversity_Loss(nn.Module):
    def __init__(self):
        super(diversity_Loss, self).__init__()

    def diversityloss(self,x1,x2,x3):
        diversity = torch.sqrt( torch.norm(x1-x2,2) + 1e-8 ) + \
                    torch.sqrt( torch.norm(x1-x3,2) + 1e-8 ) + torch.sqrt( torch.norm(x3-x2,2) + 1e-8 )
        diversity = diversity / (3 * x1.view(-1).dim())
        return diversity


    def __call__(self,x1,x2,x3):
        a = self.diversityloss(x1,x2,x3)
        return a

