import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# mode = ['train', 'test']
class MyDataset(Dataset):
    def __init__(self, path, mode, transforms):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.img_path_list, self.labels = self.get_path_list()
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        # Get image data     
        image = cv2.imread(img_path)
        image = np.transpose(image, (2,0,1))
        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.mode == 'train':
            label = self.labels[index]
            return image, label
        else: # 'test'
            return image
    
    
    def get_path_list(self):
        if self.mode == 'train':
            train_csv = pd.read_csv(self.path, index_col=0)
            img_path = train_csv['img_path'].tolist()
            labels = train_csv['labels'].tolist()
        
        if self.mode == 'test':
            test_csv = pd.read_csv(self.path, index_col=0)
            img_path = test_csv['img_path'].tolist()
            labels = None
        return img_path, labels

    

class MyTrainSetWrapper(object):
    def __init__(self, train_path, batch_size, valid_size, num_workers=0):
        self.path = train_path
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers
        
    def get_data_loaders(self):
        data_augment = self._get_train_transform()
        
        train_dataset = MyDataset(path=self.path, mode='train', transforms=data_augment)
        train_loader, valid_loader = self.get_train_validation_loaders(train_dataset)
        return train_loader, valid_loader
        
    def _get_train_transform(self):
        data_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((616,820)),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ])
        return data_transforms
    
    def get_train_validation_loaders(self, train_dataset):
        from torch.utils.data.sampler import SubsetRandomSampler
        # validation에 사용될 인덱스들
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        
        split = int(np.floor(self.valid_size*num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # train과 validation batch로 사용될 샘플
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                 num_workers=self.num_workers, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                 num_workers=self.num_workers, shuffle=False)
        
        return train_loader, valid_loader
        
class MyTestSetWrapper(object):
    def __init__(self, test_path, batch_size, num_workers=0):
        self.path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_test_loaders(self):
        data_augment = self._get_test_transform()
        
        test_dataset = MyDataset(path=self.path, mode='test', transforms=data_augment)
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=False)
        
        return test_loader
    
    def _get_test_transform(self):
        data_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((616,820)),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ])
        return data_transforms
