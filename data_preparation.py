'''
This file prepare test the new dataset
and fix train-test-split for the new dataset 
'''

from datasets_ui.dataset import UIDataset
from torch.utils.data import DataLoader

input_shape = (1280, 1920) 
category_path = "./data/categories.txt"
train_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape, is_train=True)
test_dataset = UIDataset(data_path="./data", category_path=category_path, input_shape=input_shape, is_train=False) 
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True)
eval_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True)

print(train_dataset[0].target)