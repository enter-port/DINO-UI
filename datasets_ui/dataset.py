import os 
import math
import json
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset
from util_ui.data_utils import parse_xml, image_resize, preprocess_input

class UIDataset(Dataset):
    def __init__(self, data_path, category_path, input_shape=(1080, 1080), is_train=True, split_radio=0.8):
        super(UIDataset, self).__init__()
        # self.stride = 4
        self.input_shape = input_shape
        # self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.is_train = is_train
        
        # read from .txt the categories we are training for
        # Here the categories only include four icons and level0-2
        self.category = []
        with open(category_path, 'r', encoding='utf-8') as file:
            for line in file:
                cat = line.strip()
                self.category.append(cat)
        self.num_cat = len(self.category)
        
        # if no pre-defined train-test split exists, we create a random one
        if not os.path.exists(data_path + "/train_test_split.json"):
            print("No train-test split. We create one at random.")
            datanames = []
            subject = os.listdir(data_path)
            for sub in subject:
                if not os.path.isdir(data_path + "/" + sub):
                    continue
                files = os.listdir(data_path + "/" + sub)
                for filename in files:
                    file_root, file_extension = os.path.splitext(filename)
                    print("/{}/{}".format(sub, file_root))
                    if "/{}/{}".format(sub, file_root) not in datanames:
                        datanames.append("/{}/{}".format(sub, file_root))
            random.shuffle(datanames)
            train_datanames = datanames[:int(len(datanames) * split_radio)]
            test_datanames = datanames[int(len(datanames) * split_radio):]
            data_dict = {"train": train_datanames, "test": test_datanames}
            with open(data_path + "/train_test_split.json", 'w') as f:
                json.dump(data_dict, f, indent=4)
            print("Split strored at {}.".format(data_path + "/train_test_split.json"))
        
        # else we directly read for train_test_split.json
        else:
            print("Split exists. Using split at{}".format(data_path + "/train_test_split.json"))
            with open(data_path + "/train_test_split.json", 'r') as f:
                data = json.load(f)
            train_datanames = data["train"]
            test_datanames = data["test"]
            
        # Set the files we want to read according to IS_TRAIN
        data_names = train_datanames if is_train else test_datanames
            
        # read in images, boundboxes and names; name is expressed as index of category
        self.images = []
        self.bboxes = []    # the bboxes should be a list of bbox, bbox:[[x0, y0, x1, y1, name], [...]]
        pbar = tqdm(data_names)
        for name in pbar:
            img_path = data_path + name + ".png"
            xml_path = data_path + name + ".xml"
        
            # read in .png image and convert it into desired size
            image = Image.open(img_path)
            image = np.array(image)[:, :, :3]
            
            # read in bbox and name from .xml file
            coords, names = parse_xml(xml_path)
            image, bbox = image_resize(image, input_shape, np.array(coords))
            image = preprocess_input(image)
            assert len(coords) == len(names)
            
            # add category index to bbox
            category_indices = np.array([int(self.category.index(name.lower()))+1 for name in names])
            bbox = np.column_stack((bbox, category_indices)) # (nb_num, 5)
            
            self.images.append(image)
            self.bboxes.append(bbox)
            
        self.length = len(self.images)
        assert len(self.images) == len(self.bboxes)
        assert len(self.images) == len(data_names)
        print("{} Dataset Initialize Done. {} objects in total.".format("Train" if is_train else "Test", self.length))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image = self.images[idx]
        bbox = np.array(self.bboxes[idx])
        bbox_origin = bbox
        bbox = bbox[:, :4]
        cat_ids = bbox_origin[:, 4]
        
        h, w = (image.shape)[1:]
        # w, h = (image.shape)[1:]

        bbox = torch.as_tensor(bbox, dtype=torch.float32).reshape(-1, 4)
        x0, y0, x1, y1 = bbox.unbind(-1)
        bbox = torch.stack([
            (x0 + x1) / 2 / w,  # cx
            (y0 + y1) / 2 / h,  # cy
            (x1 - x0) / w,      # w
            (y1 - y0) / h       # h
        ], dim=-1)

        # 将类别 ID 转换为张量
        labels = torch.tensor(cat_ids, dtype=torch.int64)

        # 构建目标字典
        target = {
            "boxes": bbox,
            "labels": labels,
            "image_id": torch.tensor([0]),  # 假设 image_id 为 0
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)])
        }
        return image, target
