import os
import cv2, torch, random, itertools, time,datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss, MSELoss

from Adapt_segFormer import *
from dataset import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 2023 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def train():
    base_lr = 0.0005
    ema_decay = 0.99
    num_classes = 2
    batch_size = 12
    base_dir = './data/polyp/'     # polyp  skin   idrid
    dataset = 'polyp'
    image_size = (512,512)
    consistency = 0.1
    max_epoch = 62    
   
    def create_model(image_size, num_classes,ema):
        model = Adapter_SegFormer_B4(image_size,num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
        
    model = create_model(image_size[0], num_classes,False)
    model.cuda()
    
    for name,p in model.named_parameters():
        if 'linear_pred'  in name or 'pool' in name: 
            p.requires_grad = True
        elif 'DTC' in name or 'ssf' in name or 'adaptmlp' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    model_name = 'SegB4_our_polyp'
    ema_model = create_model(image_size[0], num_classes,True)
    ema_model.cuda()
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    print('Trainable model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad == True)/1e6,'M' )
    
    db_train = semi_BaseDataSets(base_dir+'train/', 'train.txt',image_size,dataset)
    total_samples = len(db_train)
    labeled_samples = int(total_samples*0.1)
    labeled_bs = int(batch_size/2)
    print("=> Total samples is: {}, labeled samples is: {}".format(total_samples, labeled_samples))
    labeled_idxs = list(range(0, labeled_samples)) *9
    unlabeled_idxs = list(range(labeled_samples, total_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=4, pin_memory=True)
    print('=> train len:', len(train_loader))

    #optimizer = optim.Adam(model.parameters(), betas=(0.9,0.99), lr=base_lr, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(),lr=base_lr,weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    criterion_cos = CriterionCosineSimilarity()
    dice_loss = DiceLoss(num_classes)
    ssim_loss = SSIM()
    scaler = torch.cuda.amp.GradScaler()

    iter_num = 0
    max_indicator = 0
    best_IoU =0
    best_Dice =0
    best_MAE = 0
    best_Acc = 0
    tbest_IoU =0
    tbest_Dice =0
    tbest_MAE = 0
    tbest_Acc = 0
    max_iterations =  max_epoch * len(train_loader)
    for epoch_num in range(max_epoch):
        train_acc = 0
        train_loss = 0
        start_time = time.time()
        model.train()
        
        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            ema_inputs = images#[labeled_bs:]
            with torch.cuda.amp.autocast():
            #if True:
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)

                with torch.no_grad():
                    ema_output = ema_model(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)

                #ssimloss = ssim_loss(outputs_soft[:labeled_bs,1,:,:],labels[:labeled_bs].float())
                ce_losses = ce_loss(outputs[:labeled_bs],labels[:labeled_bs].long())
                dice_losses = dice_loss(outputs_soft[:labeled_bs],labels[:labeled_bs].long())
                sup_loss = 0.5*(ce_losses+dice_losses) #+ ssimloss
                consistency_loss = torch.mean((outputs_soft-ema_output_soft)**2)

                consistency_weight = get_current_consistency_weight(consistency,iter_num,max_iterations)
                loss = sup_loss  + consistency_weight*consistency_loss

                prediction = torch.max(outputs[:labeled_bs],1)[1]
                train_correct = (prediction == labels[:labeled_bs]).float().mean().cpu().numpy()
                train_acc = train_acc + train_correct
                train_loss = train_loss + loss.detach().cpu().numpy()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update() 
            update_ema_variables(model, ema_model, ema_decay, iter_num)

            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            iter_num = iter_num + 1
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        model.eval()
        print('Epoch: {} / {} '.format(epoch_num, max_epoch), 'Training time {}'.format(total_time_str),'Initial LR {:4f}'.format(lr)) 
        print('train_loss: ',train_loss/len(train_loader),' train_acc: ',train_acc/(len(train_loader)),'LR {:4f}'.format(lr)) 
        
        
        save_dir='./results/'
        db_val = testBaseDataSets(base_dir+'test/', 'test.txt',image_size,dataset)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=0)
        model.eval()
        j = 0
        if dataset == 'idrid':
            evaluator_EX= Evaluator()
            evaluator_HE= Evaluator()
            evaluator_MA= Evaluator()
            evaluator_SE= Evaluator()
            with torch.no_grad():
                for sampled_batch in valloader:
                    images, labels = sampled_batch['image'], sampled_batch['label']
                    images, labels = images.cuda(),labels.cuda()

                    pred = model(images)
                    probs = pred.cpu().numpy()
                    save_results(probs,save_dir,image_size[0],image_size[1],j)
                    j = j+1
                    predictions = torch.argmax(pred, dim=1)
                    pred = F.one_hot(predictions.long(), num_classes=num_classes)
                    new_labels =  F.one_hot(labels.long(), num_classes=num_classes)

                    evaluator_EX.update(pred[0,:,:,1], new_labels[0,:,:,1].float())
                    evaluator_HE.update(pred[0,:,:,2], new_labels[0,:,:,2].float())
                    evaluator_MA.update(pred[0,:,:,3], new_labels[0,:,:,3].float())
                    evaluator_SE.update(pred[0,:,:,4], new_labels[0,:,:,4].float())
            MAE_ex, Recall_ex, Pre_ex, Acc_ex, Dice_ex, IoU_ex = evaluator_EX.show(False)
            MAE_he, Recall_he, Pre_he, Acc_he, Dice_he, IoU_he = evaluator_HE.show(False)
            MAE_ma, Recall_ma, Pre_ma, Acc_ma, Dice_ma, IoU_ma = evaluator_MA.show(False)
            MAE_se, Recall_se, Pre_se, Acc_se, Dice_se, IoU_se = evaluator_SE.show(False)
            MAE =  (MAE_ex + MAE_he + MAE_ma +MAE_se )/4
            Acc =  (Acc_ex + Acc_he + Acc_ma +Acc_se )/4
            Dice =  (Dice_ex + Dice_he + Dice_ma +Dice_se)/4 
            IoU =  (IoU_ex + IoU_he + IoU_ma +IoU_se)/4 
            indicator = Dice+IoU  
            if indicator > max_indicator:
                best_Dice =Dice
                best_IoU =IoU
                best_MAE = MAE
                best_Acc = Acc
                max_indicator = indicator
                torch.save(model.state_dict(), './new/'+model_name+'.pth')  
            print("MAE: ", "%.2f" % MAE," Acc: ", "%.2f" % Acc," Dice: ", "%.2f" % Dice," IoU: " , "%.2f" % IoU)         

        else:
            evaluator = Evaluator()
            evaluator2 = Evaluator()
            with torch.no_grad():
                for sampled_batch in valloader:
                    images, labels = sampled_batch['image'], sampled_batch['label']
                    images, labels = images.cuda(),labels.cuda()

                    predictions  = model(images)
                    pred = predictions[0,1,:,:]
                    predictions2 = ema_model(images)
                    pred2 = predictions2[0,1,:,:]
                    evaluator.update(pred, labels[0,:,:].float())
                    evaluator2.update(pred2, labels[0,:,:].float())

                    for i in range(1):
                        #images = images[i].cpu().numpy()
                        #labels = labels.cpu().numpy()
                        #label = (labels[i]*255)
                        pred = pred.cpu().numpy()
                        #cv2.imwrite(save_dir+'image'+str(j)+'.jpg',images.transpose(1, 2, 0)[:,:,::-1])
                        #cv2.imwrite(save_dir+'GT'+str(j)+'.jpg',label*255)
                        cv2.imwrite(save_dir+'Pre'+str(j)+'.jpg',pred*255)
                        j=j+1
            MAE, Recall, Pre, Acc, Dice, IoU = evaluator.show(False)  
            ema_MAE, ema_Rec, ema_Pre, ema_Acc, ema_Dice, ema_IoU = evaluator2.show(False) 
            indicator = Dice+IoU  
            if indicator > max_indicator:
                best_Dice =Dice
                best_IoU =IoU
                best_MAE = MAE
                best_Acc = Acc
                tbest_Dice =ema_Dice
                tbest_IoU =ema_IoU
                tbest_MAE = ema_MAE
                tbest_Acc = ema_Acc
                max_indicator = indicator
                torch.save(model.state_dict(), './new/'+model_name+'.pth') 
            print('Stu: ',"MAE: ", "%.2f" % MAE," Acc: ", "%.2f" % Acc," Dice: ", "%.2f" % Dice," IoU: " , "%.2f" % IoU)
            print('Tea: ',"MAE: ", "%.2f" % ema_MAE," Acc: ", "%.2f" % ema_Acc," Dice: ", "%.2f" % ema_Dice," IoU: " , "%.2f" % ema_IoU)

    print('best student: MAE %.2f Acc %.2f Dice %.2f  IoU %.2f' %(best_MAE,best_Acc,best_Dice, best_IoU))
    print('best teacher: MAE %.2f Acc %.2f Dice %.2f  IoU %.2f' %(tbest_MAE,tbest_Acc,tbest_Dice, tbest_IoU))
    
train()