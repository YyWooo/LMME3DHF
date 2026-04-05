import os
import json
import time
import sys
sys.path.append('./')
sys.path.append('./videollama2/model')
import torch
from torch import optim
from torch.optim import lr_scheduler
import shutil
from opts import parse_opts
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from sal_losses import SalLoss

from torch.optim import Adam
from utils import Logger
from dataset import get_training_set, get_validation_set, get_test_set

from visualize_loss import loss_visualize
from train import train_epoch
from validation import val_epoch
import test_sal
from datetime import datetime
import random
import numpy as np

from torch.cuda.amp import GradScaler

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    opt = parse_opts()
    if opt.precision == 'f32':
        precision = 'full'
    elif opt.precision == 'f16':
        precision = 'half'
    elif opt.precision == 'bf16':
        precision = 'bf'
    elif opt.precision == 'ori':
        precision = 'ori'
        
    if opt.mm_train:
        precision = precision + '_mm'   

    if opt.backbone == "VideoLLaMA2.1-7B-16F":
        backbone = "Qwen"
        opt.sample_num = 16
        from temporal_transforms_Qwen import TemporalRandomCrop, TemporalStartCrop, TemporalCenterCrop

    if opt.no_train and opt.test or opt.model_name == "DEBUG":
        if opt.instruct == '':
            opt.instruct = 'What is the salient region in this image?'
        
    if opt.model_name != "DEBUG" and not opt.no_train:
        if opt.instruct == '':
            opt.instruct = 'What is the salient region in this image?'
        else:
            precision = precision + "_prompt"
        opt.model_name = opt.dataset + '_' + datetime.now().strftime("%Y%m%d_%H%M") + '_' + opt.optimizer + '_d' + str(opt.lr_decoder) + '_p' + str(opt.lr_projector) + opt.scheduler + '_' + opt.ver + '_s' + str(opt.manual_seed) + '_bs' + str(opt.update_step) + backbone[0] + '_b' + str(opt.b) + '_' + precision
    disable_torch_init()
    
    os.makedirs(os.path.join(opt.result_path, opt.model_name), exist_ok=True)
    print(f"Result path: {os.path.join(opt.result_path, opt.model_name)}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt.manual_seed)
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)
    
    with open(os.path.join(opt.result_path, opt.model_name, 'opts' + '_test' + str(opt.test) + datetime.now().strftime('%Y%m%d_%H%M%S') +'.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
        

    model_path = os.path.join('DAMO-NLP-SG', opt.backbone)
    if opt.precision in ['bf16', 'ori']:
        model, processor, tokenizer = model_init(model_path, torch_dtype=torch.bfloat16)
    elif opt.precision in ['f16', 'f32']:
        model, processor, tokenizer = model_init(model_path)
    
    for param in model.parameters():
        param.requires_grad = False
    if not opt.mm_train:
        for param in model.model.mm_projector.readout.parameters():
            param.requires_grad = True
    else:
        for param in model.model.mm_projector.parameters():
            param.requires_grad = True

    # 在这里接一个decoder
    module_path = f"videollama2.model.saliency_decoder_{backbone}"
    decoder_class = getattr(__import__(module_path, fromlist=["SaliencyDecoder"]), "SaliencyDecoder")
    if opt.backbone == "VideoLLaMA2.1-7B-16F":
        decoder = decoder_class(feature_dim=3584)
    
    if opt.precision == 'f16':
        decoder = decoder.to(torch.float16)
    elif opt.precision == 'bf16':
        decoder = decoder.to(torch.bfloat16)

    decoder.to(device)
    
    if opt.resume_path:
        print('Loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        opt.begin_epoch = checkpoint['epoch']
        if not opt.mm_train:
            model.model.mm_projector.readout.load_state_dict(checkpoint['readout_state_dict'])
        else:
            model.model.mm_projector.load_state_dict(checkpoint['mm_projector_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
    if not opt.no_train:
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        optimizer= optim.SGD( 
            [
                {'params': model.model.mm_projector.parameters(), 'lr': opt.lr_projector},
                {'params': decoder.parameters(), 'lr': opt.lr_decoder}
            ],
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
            
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=0.0001)
       

        criterion = SalLoss()
        temporal_transform = TemporalRandomCrop(opt)
        
        with open(os.path.join(opt.result_path, opt.model_name, 'model.txt'), 'w') as f:
                print(decoder, file=f)
                print(criterion, file=f) 
                print(scheduler, file=f) 
        

        training_data = get_training_set(opt, processor, temporal_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            drop_last=True,
            pin_memory=False
        )
        
        train_log = os.path.join(opt.result_path, opt.model_name, 'train_epoch.log')
        train_logger = Logger(
            train_log, ['epoch', 'epoch_time', 'loss', 'nll', 'CC', 'KL', 'BCE', 'lr_d', 'lr_p'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, opt.model_name, 'train_batch.log'), ['epoch', 'batch', 'loss', 'nll_loss', 'CC_loss', 'KL_loss', 'BCE_loss', 'lr_d', 'lr_p'])
        
    if not opt.no_val:
        temporal_transform = TemporalCenterCrop(opt)
 
        if opt.dataset in ['SALICON', 'MIT1003', 'UCF', 'Hollywood2', 'DHF1K', 'MINE']: 
            validation_data = get_validation_set(opt, processor, temporal_transform)
 
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            drop_last=True,
            pin_memory=True
        )
        val_log = os.path.join(opt.result_path, opt.model_name, 'val_'+opt.model_name+'.log')
        val_logger = Logger(
            val_log, ['epoch', 'loss', 'nll', 'CC', 'KL', 'BCE', 'lr_d', 'lr_p'])
       
        
    print('Running...')
    val_loss_min = 100
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            opt = train_epoch(i, opt.n_epochs, train_loader, model, decoder, tokenizer, optimizer, criterion, opt, train_logger, train_batch_logger)         
            if not opt.no_val:
                os.makedirs(opt.val_log_path, exist_ok=True)
                shutil.copy(os.path.join(opt.result_path, opt.model_name, 'val_'+opt.model_name+'.log'), os.path.join(opt.val_log_path, opt.model_name + '.log'))

        if not opt.no_val:
            validation_loss, val_loss_min = val_epoch(i, opt.n_epochs, val_loader, model, decoder, tokenizer, optimizer, criterion, opt, val_loss_min, val_logger)
            loss_visualize(train_log, val_log, os.path.join(opt.result_path, opt.model_name), opt.model_name)
        
        if not opt.no_train:
            scheduler.step()
       
       
    if opt.test:
        temporal_transform =  TemporalStartCrop(opt)

        test_data = get_test_set(opt, processor, temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test_sal.test(test_loader, model, decoder, tokenizer, opt)
