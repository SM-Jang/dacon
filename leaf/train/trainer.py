import torch
import torch.nn as nn
import torch.optim as optim

import os
from datetime import datetime
import numpy as np
import shutil
from model.models import Model
import matplotlib.pyplot as plt


# mymodel.yaml 파일을 모델 가중치와 세트로 저장
def _save_config_file(model_checkpoints_folder, model_name):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    shutil.copy(f'./config/{model_name}.yaml', os.path.join(model_checkpoints_folder, model_name+'.yaml'))
    
# 최초, 또는 가장 최근 학습된 가중치와 yaml 파일은 ./weight/checkpoint 폴더에 저장
# 이후에 이뤄지는 실험은 전에 저장된 checkpoint 이하 파일들은 ./weight/experiments 이하로 timestamp와 함께 copy
def _copy_to_experiment_dir(model_checkpoints_folder, model_name):
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    new_exp_dir = os.path.join('./weights/experiments', model_name + '_checkpoints', now_time)
    if not os.path.exist(new_exp_dir):
        os.makedirs(new_exp_dir)
    for src in os.listdir(model_checkpoints_folder):
        shutil.copy(os.path.join(model_checkpoints_folder, src), new_exp_dir)
        
        
        
# train.py
class Trainer(object):
    def __init__(self, dataset, base_model, config):
        self.dataset = dataset
        self.base_model = base_model
        self.config = config # yaml
        self.device = self._get_device()
        self.loss = nn.L1Loss()
        self.model_dict = {'resnet34': Model}
    
    
    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('The device type is', device)
        return device


    def _get_model(self):
        try:
            # yaml에서 설정한 모델명 오타 확인
            model = self.model_dict[self.base_model]
            return model
        except:
            raise('Invalid model name. Pass one of the model idctionary.')
        
        
    # train or inference에서 모델 가중치를 불러오기 위한 함수
    # mymodel.yaml에서 resume이 None일 경우, 가장 최신 checkpoint를 불러옴
    # 이전 모델을 불러오고 싶은 경우, 해당 폴더명(timestamp)을 resume에 설정
    def _load_pre_trained_weights(self, model):
        best_mae=9999
        start_epoch=0
        if self.config['resume'] is not None:
            try:
                checkpoints_folder = os.path.join('./weights/experiments', str(self.base_model)+'_checkpoints')
                checkpoint = torch.load(os.path.join(checkpoints_folder, self.config['resume'], 'model.pth'))
                model.load_state_dict(checkpoint['net'])
                best_mae = checkpoint['mae']
                start_epoch = checkpoint['epoch']
                print('\n==> Resuming from checkpoint..')
            except FileNotFoundError:
                print('\nPre-trained weights not found. Training from scratch')
        else:
            print('\nTrainin from scratch')
        return model, best_mae, start_epoch
    
    
    # train loop 안에서 epoch 마다 실행 될 validation
    def _validate(self, epoch, net, criterion, valid_loader, best_mae):
        net.eval()
        
        # validation steps
        valid_losses = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.float().to(self.device)
                labels = labels.float().to(self.device)
                
                # forward
                logit = net(images)
                valid_loss = criterion(logit.squeeze(1), labels)
                valid_losses.append(valid_loss.item())
                
        valid_mae = np.mean(valid_losses)
        
        # Save checkpoint
        model_checkpoints_folder = os.path.join('./weights','checkpoints')
        
        # 해당 epoch에서 best mae일 때, epoch 모델을 저장
        if valid_mae < best_mae:
            print('Saving..')
            state = {
                'net':net.state_dict(),
                'valid_mae':valid_mae,
                'epoch':epoch
            }
            torch.save(state, os.path.join(model_checkpoints_folder, 'model.pth'))
            best_mae = valid_mae
        # 다음 epoch를 위해 다시 .train() 모드로 전환
        net.train()
        
        return valid_mae, best_mae
    
    
    def train(self):
        # train.py의 main 함수에서 데이터를 가져옴
        train_loader, test_loader = self.dataset.get_data_loaders()
        
        # model 선언
        model = self._get_model()
        model = model(**self.config['model'])
        model, best_mae, start_epoch = self._load_pre_trained_weights(model)
        model = model.to(self.device)
        model.train()
        
        # Loss & optimizer
        criterion = self.loss.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader, eta_min=0, last_epoch=-1))
        
        # Save config file
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints')
        _save_config_file(model_checkpoints_folder, str(self.base_model))
        
        history = {}
        history['train_loss'] = []
        history['valid_loss'] = []
        
        num_epochs = self.config['epochs']
        
        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_losses = []
            train_loss = 0
            for i, (images, labels) in enumerate(train_loader, 0):
                # CPU -> GPU
                images = images.float().to(self.device)
                labels = labels.float().to(self.device)
                
                # forward
                logits = model(images)
                loss = criterion(logits.squeeze(1), labels)
                train_loss+=loss.item()
                train_losses.append(loss.item())
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f'Running [{i+1}/{len(train_loader)}] Loss {train_loss:.4f}')
            
            train_loss = np.mean(train_losses)
            valid_loss, best_mae = self._validate(epoch, model, criterion, test_loader, best_mae)
            print('EPOCHS [{}/{}] train_loss:{:.4f} valid_loss:{:.4f}'.format(epoch+1, start_epoch+num_epochs, train_loss, valid_loss))
            
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)

            # plt.figure(figsize=(15, 10))
            # plt.plot(history['train_loss'], linewidth=2.0)
            # plt.plot(history['valid_loss'], linewidth=2.0)
            # plt.title('model loss.')
            # plt.ylabel('loss')
            # plt.xlabel('epoch')
            # plt.legend(['train', 'valid'], loc='upper right')
            # plt.savefig('./weights/checkpoints/loss.png')
            # plt.close()
    
        # copy and save trained model with config to experiments dir.
        _copy_to_experiment_dir(model_checkpoints_folder, str(self.base_model))

        print("--------------")
        print("All files saved.")
        
    