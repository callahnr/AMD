import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode as CN
import sys
from datetime import datetime
import argparse

import config
import data
import logs
import model
import optimizers
import criterion
from test import test
from train import train
import transforms


#   BUILD MODEL FROM CONFIG
cfg = config.default_cfg()
if __name__ == '__main__':
    cfg.merge_from_list(sys.argv[1:])
make_net = getattr(model, cfg.model)
make_optm = getattr(optimizers, cfg.optimizer)
make_loss_fn = getattr(criterion, cfg.loss_fn)
make_transform = getattr(transforms, cfg.transform)

net = make_net()
optimizer = make_optm(net.parameters(), cfg.learning_rate)
loss_fn = make_loss_fn()
transform = make_transform()


#   DATA SETUP
print("Loading training data set.")
trainset = data.DeepSig2018(split="train")
print("Loading testing data set.")
testset = data.DeepSig2018(split="test")
print("")


#   LOGGER AND TENSORBOARD SETUP
save_dir, log_file, chkpt_dir = logs.setup_logs(cfg)


#   GPU SETUP
print("Checking for GPU's")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

print(logs.log_header(cfg, log_file))
writer = SummaryWriter(cfg.writer_file, comment=cfg.dump())

#    CURRICULUM LOOP
snr_lvls = range(30, -20, -2)
for lesson, min_snr in enumerate(snr_lvls):
    best_epoch = 0
    epoch_since_best = 0
    lesson_train_accu = []
    lesson_train_losses = []
    lesson_eval_accu = []
    lesson_eval_losses = []
    

    if lesson >=1:
        net.load_state_dict(torch.load(f'{chkpt_dir}best_model-minSNR{min_snr+2}_maxSNR30.pt'))

    print(f"Filtering data to minSNR:{min_snr} to maxSNR:{cfg.max_snr}.")
    trainset.filter_SNR(min_snr=min_snr, max_snr=cfg.max_snr)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=(not cfg.micro), num_workers=0)
    testset.filter_SNR(min_snr=min_snr, max_snr=cfg.max_snr)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    #    TRAINING/EVAL LOOP
    print(f"Begin Training Lesson {lesson+1}")
    for epoch in range(1, cfg.epochs+1):
        print("training...")
        train_losses, train_accu = train(trainloader, optimizer, loss_fn, net, device, micro=cfg.micro)
        print("testing...")
        eval_losses, eval_accu = test(testloader, loss_fn, net, device, micro=cfg.micro) 

        lesson_train_accu.append(train_accu)
        lesson_train_losses.append(train_losses)

        lesson_eval_accu.append(eval_accu)
        lesson_eval_losses.append(eval_losses)
        
        #   CHECKPOINTS AND EARLY STOPPING
        if epoch > 1:
            if lesson_eval_accu[-1] == max(lesson_eval_accu):
                chkpt_file = f'best_model-minSNR{min_snr}_maxSNR{cfg.max_snr}.pt'
                
                torch.save(net.state_dict(), chkpt_dir+chkpt_file)
                best_epoch = epoch
                epoch_since_best = 0
            else:
                epoch_since_best += 1

        if epoch_since_best >= cfg.early_stop:
            print(f"No improvement for {cfg.early_stop} epochs. Ending Lessons.")
            break
        
        #   LOGGER AND TENSORBOARD
        max_accu  = max(lesson_eval_accu)
        logs.log_results(cfg, train_losses, train_accu, 
                            eval_losses, eval_accu, 
                            epoch, writer, log_file,
                            lesson, min_snr, max_accu,
                            best_epoch, epoch_since_best)
    
    #   END TRAINING/EVAL LOOP

    logs.log_lesson(lesson+1, log_file)
#   END CURRICULUM LOOP