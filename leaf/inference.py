import yaml
import torch
import numpy as np
import os
import pandas as pd
from utils.dataloader import MyTestSetWrapper
from model.models import Model

## load model
def _get_model(base_model):
    model_dict = {'resnet34':Model}
    
    try:
        model= model_dict[base_model]
        return model
    except:
        raise "Invalid model name. Pass one of the momdel dictionary."
        
def _load_weights(model, load_from, base_model):
    try:
        checkpoints_folder = os.path.join('./weights/experiments', str(base_model)+'_checkpoints')
        checkpoint = torch.load(os.path.join(checkpoints_folder, load_from, 'model.pth'))
        model.load_state_dict(checkpoint(['net']))
        print('\n==> Resuming from checkpoint..')
        
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

def _load_weights_from_recent(model):
    try:
        checkpoints_folder = os.path.join('./weights', 'checkpoints')
        checkpoint = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        model.load_state_dict(checkpoint['net'])
        print('\n==> Resuming from checkpoint..')
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

## main
def main(model_name):
    checkpoints_folder = os.path.join('./weights', 'checkpoints')
    print(os.listdir(checkpoints_folder))
    config = yaml.load(open(checkpoints_folder + '/' + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device type is:', device)
    
    ## get class names
    testset = MyTestSetWrapper(**config['test'])
    
    ## model load
    # model topology
    model = _get_model(model_name)
    model = model(**config['model'])
    
    # model weight
    
    if config['resume'] != "None":
        model = _load_weights(model, config['resume'], model_name)
        model = model.to(device)
    else:
        model = _load_weights_from_recent(model)
        model = model.to(device)
    model.eval()
    
    ## test_loader
    test_loader = testset.get_test_loaders()
    
    
    # inference
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.float().to(device)
            
            # calculate outputs by running images through the network
            
            prediction = model(images)
            predictions.extend(prediction.detach().cpu().numpy().squeeze().tolist())
            
    submission = pd.read_csv('./dataset/sample_submission.csv')
    submission['leaf_weight'] = predictions
    submission.to_csv('./submit.csv', index=False)
    print("Done!")
    
if __name__ == '__main__':
    model_name='resnet34'
    main(model_name)