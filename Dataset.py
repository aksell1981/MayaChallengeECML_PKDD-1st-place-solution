import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import KFold
import pandas as pd
import os
import torch
from albumentations import *

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class MayaDataset(Dataset):
     
    def __init__(self, data_path, transform=None,prefix=None,
                 train=True,nfolds=4,fold=0):
        fns=os.listdir(data_path+'lidar_train')
        ids=[ int(fn.split('_')[1] )for fn in fns]
        df=pd.DataFrame({'fn':fns,'id':ids})
        ids=df.id.values
        kf = KFold(n_splits=nfolds,random_state=42,shuffle=True)
        ids = list(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.img_dir = data_path
        self.transform = transform      
        self.prefix=prefix
        self.ids=ids
        print('prefix:',self.prefix,'len=',len(self.ids),'train=',train,'fold=',fold)

    def __getitem__(self, index):
        id=self.ids[index]
        fn = 'tile_'+str(id)+'_lidar.tif'  
        image=cv2.imread(os.path.join(self.img_dir,'lidar_train', fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fnm='tile_'+str(id)+'_mask_'+ self.prefix +'.tif'
        mask=cv2.imread(os.path.join(self.img_dir,'train_masks', fnm),cv2.IMREAD_GRAYSCALE)
        mask=mask/255
        mask=np.logical_not(mask).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask= augmented['mask']
        data={'img':img2tensor((image/255.0 - mean)/std), 'mask':img2tensor(mask)}
        return data
        
    def __len__(self):
        return len(self.ids)




def get_aug2(p=1.0):
     return Compose([
         HorizontalFlip(),
         VerticalFlip(),
         RandomRotate90(),
         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5,
                          border_mode=cv2.BORDER_REFLECT),
         RandomBrightnessContrast(p=0.5),
    HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),

    
    OneOf([
        OpticalDistortion(p=0.3),
        GridDistortion(p=.1),
        IAAPiecewiseAffine(p=0.3),
    ], p=0.3),
    
     ], p=p)
def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            ElasticTransform(p=.3),
            GaussianBlur(p=.3),
            GaussNoise(p=.3),
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(15,25,0),
            #CLAHE(clip_limit=2),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        ], p=0.3),
    ], p=p)

    #break
