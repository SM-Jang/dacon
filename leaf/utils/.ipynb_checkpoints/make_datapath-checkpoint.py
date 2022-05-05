import pandas as pd
import os
from glob import glob

def make_trainpath_csv(data_dir):
    img_path_list = []
    label_list = []
    for case_name in os.listdir(data_dir):
        current_path = os.path.join(data_dir, case_name)
        if os.path.isdir(current_path):
            # get image path
            img_path_list.extend(glob(os.path.join(current_path, 'image', '*.jpg')))
            img_path_list.extend(glob(os.path.join(current_path, 'image', '*.png')))
            
            # get label
            label_df = pd.read_csv(current_path+'/label.csv')
            label_list.extend(label_df['leaf_weight'])
                
    # Make csv File
    train_data = {
    'img_path':img_path_list,
    'labels':label_list
    }
    train_data = pd.DataFrame(train_data)
    train_data.to_csv(data_dir+'.csv')
    

def make_testpath_csv(data_dir):
    # get image path
    img_path_list = glob(os.path.join(data_dir, 'image', '*.jpg'))
    img_path_list.extend(glob(os.path.join(data_dir, 'image', '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    
    # Make csv File
    test_data = {
        'img_path':img_path_list
    }
    test_data = pd.DataFrame(test_data)
    test_data.to_csv(data_dir+'.csv')
    
    
if __name__=='__main__':
    make_trainpath_csv('./dataset/train')
    make_testpath_csv('./dataset/test')