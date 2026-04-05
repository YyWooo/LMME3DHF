import torch
import torch.nn as nn
# from taming.modules.losses.lpips import LPIPS

class CCLoss(nn.Module):
    def __init__(self):
        super(CCLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8

    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # calculating the mean value of tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number
        # print("map_gtd_mean is :", map_gtd_mean)

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 
        map_gtd_std = torch.std(map_gtd) # calculate the standard deviation
        map_gtd_std = map_gtd_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization
        map_gtd = (map_gtd - map_gtd_mean) / (map_gtd_std + self.epsilon) # normalization

        map_pred_mean = torch.mean(map_pred) # re-calculating the mean value of normalized tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # re-calculating the mean value of normalized tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number

        CC_1 = torch.sum( (map_pred - map_pred_mean) * (map_gtd - map_gtd_mean) )
        CC_2 = torch.rsqrt(torch.sum(torch.pow(map_pred - map_pred_mean, 2))+ self.epsilon) * torch.rsqrt(torch.sum(torch.pow(map_gtd - map_gtd_mean, 2))+ self.epsilon) 
        CC = CC_1 * CC_2
        # print("CC loss is :", CC)
        CC = -CC # the bigger CC, the better

        # we put the L1 loss with CC together for avoiding building a new class
        # L1_loss =  torch.mean( torch.abs(map_pred - map_gtd) )
        # print("CC and L1 are :", CC, L1_loss)
        # CC = CC + L1_loss
        if torch.isnan(CC):
            print('CC_1',CC_1)
            print('CC_2',CC_2)
            print('torch.sum(torch.pow(map_pred - map_pred_mean, 2))',torch.sum(torch.pow(map_pred - map_pred_mean, 2)))
            print('torch.sum(torch.pow(map_gtd - map_gtd_mean, 2))',torch.sum(torch.pow(map_gtd - map_gtd_mean, 2)))
        return CC
    

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are :", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (torch.sum(map_pred) + self.epsilon)# normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (torch.sum(map_gtd) + self.epsilon) # normalization step to make sure that the map_gtd sum to 1
        # print("map_pred is :", map_pred)
        # print("map_gtd is :", map_gtd)


        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        # print("KL 1 is :", KL)
        KL = map_gtd * KL
        # print("KL 2 is :", KL)
        KL = torch.sum(KL)
        # print("KL 3 is :", KL)
        # print("KL loss is :", KL)

        return KL
    
class SalLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0):
        super().__init__()

        # self.perceptual_loss = LPIPS().eval()
        # self.perceptual_weight = perceptual_weight

        self.CC = CCLoss()
        self.KL = KLLoss()
        self.BCE = nn.BCELoss()

    def forward(self, reconstructions, gts, split="train"):
        
        rec_loss = torch.abs(gts.contiguous() - reconstructions.contiguous())
        # if self.perceptual_weight > 0:
        #     p_loss = self.perceptual_loss(gts.contiguous(), reconstructions.contiguous())
        #     rec_loss = rec_loss + self.perceptual_weight * p_loss
        # else:
        #     p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # saliency loss
        reconstructions = torch.clamp(reconstructions, 0.0, 1.0)
        gts = torch.clamp(gts, 0.0, 1.0)
        CC_loss = self.CC(reconstructions,gts)
        KL_loss = self.KL(reconstructions,gts)
        BCE_loss = self.BCE(reconstructions,gts)

        # paired image-to-image translation loss (with condition):
        # ########################################################
        loss = nll_loss + 0.4*(1+CC_loss+0.5*KL_loss) + BCE_loss
        nll_loss = nll_loss.detach().mean()
        rec_loss = rec_loss.detach().mean()
        CC_loss = CC_loss.detach().mean()
        KL_loss = KL_loss.detach().mean()

        return loss, nll_loss, CC_loss, KL_loss, BCE_loss
