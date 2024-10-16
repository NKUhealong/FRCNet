import os
import cv2
import math
import time
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from scipy import ndimage
import torch.nn.functional as F
#from sklearn.metrics import *
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()       
        self.IoU_polyp = list()

    def evaluate(self, pred, gt):
        
        pred_binary = (pred >= 0.5).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()

        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        
        MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
        TP = pred_binary.mul(gt_binary).sum().cuda(0)
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda(0)
        # recall
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        #Specificity = TN / (TN + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)

        return MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(), Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(), Dice.data.cpu().numpy().squeeze(), IoU_polyp.data.cpu().numpy().squeeze()

        
    def update(self, pred, gt):
        mae, recall, precision, accuracy, dice, ioU_polyp = self.evaluate(pred, gt)        
        self.MAE.append(mae)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)       
        self.IoU_polyp.append(ioU_polyp)

    def show(self,flag = True):
        if flag == True:
            print("MAE: ", "%.2f" % (np.mean(self.MAE)*100),
                  " Recall: ", "%.2f" % (np.mean(self.Recall)*100), 
                  " Pre: ", "%.2f" % (np.mean(self.Precision)*100),
                  " Acc: ", "%.2f" % (np.mean(self.Accuracy)*100),
                  " Dice: ", "%.2f" % (np.mean(self.Dice)*100),
                  " IoU: ", "%.2f" % (np.mean(self.IoU_polyp)*100),'\n')
        
        return np.mean(self.MAE)*100,np.mean(self.Recall)*100,np.mean(self.Precision)*100,np.mean(self.Accuracy)*100,np.mean(self.Dice)*100,np.mean(self.IoU_polyp)*100
        
def sigmoid_rampup(current, rampup_length):
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(consistency,current_step, total_steps):
    phase = 1.0 - current_step / total_steps
    final_consistency = consistency* np.exp(-5.0 * phase * phase)
    return final_consistency 

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def compute_sdf(img_gt, out_shape):
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): 
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)+1e-8) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)+1e-8)
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
    return normalized_sdf
    
def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    #mse_loss = torch.mean ((input_softmax-target_softmax)**2)
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def kl_loss_compute(p2, p1):
    KL_loss = torch.mean(p2*torch.log(1e-8 + p2/(p1+1e-8)))
    return KL_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        #print(input_tensor.size())
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs.size(),target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice)
            loss += dice * weight[i]
        return loss / self.n_classes


class CriterionCosineSimilarity(nn.Module):
    def __init__(self):
        super(CriterionCosineSimilarity, self).__init__()
        self.ep = 1e-6

    def forward(self, p, q):
        
        sim_matrix = p.transpose(-2, -1).matmul(q)
        #sim_matrix = F.softmax(sim_matrix, dim=1)
        #a = torch.norm(p, p=2, dim=-2)
        #print(p.shape,a.unsqueeze(-2).shape)
        #sim_matrix /= a.unsqueeze(-2)
        return sim_matrix
        
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
 
        self.refl = nn.ReflectionPad2d(1)
 
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
 
    def forward(self, x, y):
        #x = self.refl(x)
        #y = self.refl(y)
 
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
 
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
 
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
 
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()      

        
def save_results(probs,save_dir,h,w,i):
    probs = probs[0] 
    pred = np.argmax(probs,axis=0)
    pred_vis = np.zeros((h,w,3),np.uint8)
    pred_vis[pred==1]=[255,0,0]
    pred_vis[pred==2]=[0,255,0]
    pred_vis[pred==3]=[0,0,255]
    pred_vis[pred==4]=[255,0,255]
    cv2.imwrite(save_dir+'Pred'+str(i)+'.png',pred_vis[:,:,::-1])