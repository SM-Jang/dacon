import os 
import pandas as pd
import numpy as np
from utils import dataloader
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_model_summary
from model.models import UNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device type is:', device)
config = {
    'epochs': 50,
    'resume': None,
    'learning_rate': 0.001
}
config['train'] = {
        'train_path':'dataset/train.csv',
        'batch_size':3,
        'valid_size':0,
        'num_workers':0
}

# Dataset, DataLoader
trainset = dataloader.MyTrainSetWrapper(**config['train'])
train_loader, valid_loader = trainset.get_data_loaders()

# model
model = UNet(3, 3)
print(pytorch_model_summary.summary(model, torch.zeros(1,3,256,256), show_input=True))
model = UNet(3, 3).to(device)

# Loss & Optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

num_epochs = config['epochs']

print('Learning Start')
model.train()
for epoch in range(num_epochs):
    train_losses = []
    train_loss = 0
    for i, (images, _) in enumerate(train_loader):
        # CPU -> GPU
        images = images.float().to(device)
        
        # forward
        fake_images = model(images)
        loss = criterion(fake_images, images)
        train_loss+=loss.item()
        train_losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Running [{i+1}/{len(train_loader)}] Loss {train_loss:.4f}')
        
    train_loss = np.mean(train_losses)
    print('EPOCHS [{}/{}] train_loss:{:.4f}'.format(epoch+1, start_epoch+num_epochs, train_loss))
    
    
PATH = './weights/'
torch.save(model.state_dict(), PATH + 'unet.pt')  # 모델 객체의 state_dict 저장
