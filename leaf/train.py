import yaml

from train.trainer import Trainer
from utils.dataloader import MyTrainSetWrapper

def main(model_name):
    # yaml
    config = yaml.load(open('config/'+str(model_name) + '.yaml', 'r'), Loader=yaml.FullLoader)
    trainset = MyTrainSetWrapper(**config['train'])
    
    # Train
    trainer = Trainer(trainset, model_name, config)
    trainer.train()
    
    
if __name__ == '__main__':
    main('resnet34')