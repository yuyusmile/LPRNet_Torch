import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import *
from label_code import *
import random
import os, cv2


class Load_Date():
    
    def __init__(self, img_file_path, img_size):
        
        self.img_file_path = img_file_path
        self.img_path = [os.path.join(self.img_file_path, name) for name in os.listdir(self.img_file_path)]
        random.shuffle(self.img_path)
        self.img_size = img_size
        # self.transopse = transopse(self, img)
     
    def transopse(self, img):
        
        img = img.astype('float32')
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        return img     
        
        
    def __len__(self):
        
        return len(self.img_path)
    
    
    def __getitem__(self, index):
        
        filename = self.img_path[index]
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        if h != self.img_size[1] or w != self.img_size[0]:
            img = cv2.resize(img, self.img_size)
        img = self.transopse(img)
        
        labels = list()
        basename = os.path.basename(filename)
        name, suffix = os.path.splitext(basename)
        name0 = name.split('_')[0]
        
        if '\u4e00' <= name0 <= '\u9fff':
            label = name[:7]
        else:
            label = dict[name[:3]] + name[4:10]
            
        encode_label = label_encode(label)
        labels.append(encode_label)
        
        len_labels = []
        for i in range(len(labels)):
            len_labels.append(len(labels[i]))
    
        return img, labels, len_labels
    
        
        
def collate_fn(batch):
    
    img_list = []
    labels_list = []
    lengts_list = []
    
    for _, sample in enumerate(batch):
        
        img, labels, len_label = sample
        img_list.append(torch.from_numpy(img))
        labels_list.extend(labels)
        lengts_list.append(len_label[0])
    
    # labels_list = np.asarray(labels_list).flatten().astype(np.float32)
    labels_list = np.asarray(labels_list).astype(np.float32)
        
    return (torch.stack(img_list, 0), torch.from_numpy(labels_list), lengts_list)

"""if __name__ == "__main__":
    
    dataset = Load_Date('./valid', (94, 24))
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('data length is {}'.format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))      
        break    
"""

    
