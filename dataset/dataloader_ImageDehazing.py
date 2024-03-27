import torch
import os
from PIL import Image
import random
from torchvision.transforms import functional as FF
import torchvision.transforms as tfs
import torch.utils.data as data
import torchvision.transforms as tt
from dataset_path_config import get_path_dict_ImageDehazing


def pair_flip(img, target):
    if random.random() > 0.5:
        img = FF.hflip(img)
        target = FF.hflip(target)
    if random.random() > 0.5:
        img = FF.vflip(img)
        target = FF.vflip(target)

    return img, target


def pair_crop(img, target, new_size):
    # if random.random() > 0.5 and img.size()[1] >= new_size.crop_size[0] and img.size()[2] >= new_size[1]:
    i, j, h, w = tfs.RandomCrop.get_params(img, output_size=(new_size[0], new_size[1]))
    img = FF.crop(img, i, j, h, w)
    target = FF.crop(target, i, j, h, w)
    return img, target


class Dataset_OHAZE(data.Dataset):
    def __init__(self, path, img_size, if_flip, if_crop, crop_size, trans_hazy=None, trans_gt=None):
        super(Dataset_OHAZE, self).__init__()

        self.haze_imgs_dir = os.listdir(os.path.join(path, "hazy/"))
        self.haze_imgs = [os.path.join(path, "hazy/", img) for img in self.haze_imgs_dir]

        self.clear_dir = os.path.join(path, "clear/")

        self.img_size = img_size
        self.if_flip = if_flip
        self.if_crop = if_crop
        self.crop_size = crop_size

        self.trans_hazy = None
        self.trans_gt = None
        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                     tt.ToTensor()])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                        tt.ToTensor()])

        self.split = "/"
        # if platform.system() == "Windows":
        #     self.split = "\\"

    def __getitem__(self, index):

        data_hazy = Image.open(self.haze_imgs[index]).convert('RGB')
        hazy_img = self.haze_imgs[index]
        clear_name = hazy_img.split(self.split)[-1]
        data_gt = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        if self.if_flip:
            data_hazy, data_gt = pair_flip(data_hazy, data_gt)

        if self.if_crop:
            data_hazy, data_gt = pair_crop(data_hazy, data_gt, self.crop_size)

        data_hazy = self.trans_hazy(data_hazy)
        data_gt = self.trans_gt(data_gt)

        tar_data = {"hazy": data_hazy,
                    "gt": data_gt,
                    "name": hazy_img.split(self.split)[-1],
                    "hazy_path": self.haze_imgs[index]}

        return tar_data

    def __len__(self):
        return len(self.haze_imgs)


class Dataset_OHAZE_val_split(data.Dataset):
    def __init__(self, pred_path, label_path, img_size, trans_hazy=None, trans_gt=None):
        super(Dataset_OHAZE_val_split, self).__init__()

        self.haze_imgs_dir = os.listdir(pred_path)
        self.haze_imgs = [os.path.join(pred_path, img) for img in self.haze_imgs_dir]

        self.clear_dir = label_path

        self.img_size = img_size

        self.trans_hazy = None
        self.trans_gt = None
        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                     tt.ToTensor()])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                        tt.ToTensor()])

        self.split = "/"

    def __getitem__(self, index):
        data_hazy = Image.open(self.haze_imgs[index]).convert('RGB')
        hazy_img = self.haze_imgs[index]
        clear_name = hazy_img.split(self.split)[-1]
        data_gt = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        data_hazy = self.trans_hazy(data_hazy)
        data_gt = self.trans_gt(data_gt)
        tar_data = {"hazy": data_hazy,
                    "gt": data_gt,
                    "name": hazy_img.split(self.split)[-1],
                    "hazy_path": self.haze_imgs[index]}

        return tar_data

    def __len__(self):
        return len(self.haze_imgs)



def get_train_val_loader(dataset, img_h, img_w, train_batch_size, num_workers, if_flip, if_crop, crop_h, crop_w):
    supported_dataset = {
        "NHR": Dataset_OHAZE,
        "NHM": Dataset_OHAZE,
        "NHCL": Dataset_OHAZE,
        "NHCM": Dataset_OHAZE,
        "NHCD": Dataset_OHAZE,
        "UNREAL_NH": Dataset_OHAZE,
        "UNREAL_NH_NoSky_Dark": Dataset_OHAZE,
        "GTA5": Dataset_OHAZE,
        "NightHaze": Dataset_OHAZE,
        "YellowHaze": Dataset_OHAZE,
        "RWNHC_MM23_PseudoLabel": Dataset_OHAZE,
        "RWNHC_MM23_PseudoLabel_mini": Dataset_OHAZE
    }

    path_dict = get_path_dict_ImageDehazing()

    try:
        data_root_train = path_dict[dataset]["train"]
        data_root_val = path_dict[dataset]["val"]

    except:
        raise ValueError("dataset not support")
    img_size = [img_h, img_w]
    crop_size = [crop_h, crop_w]

    train_dataset = supported_dataset[dataset](data_root_train, img_size=img_size,
                                               if_flip=if_flip, if_crop=if_crop, crop_size=crop_size)

    val_dataset = supported_dataset[dataset](data_root_val, img_size=img_size,
                                             if_flip=False, if_crop=False, crop_size=None)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=train_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True,
                                             drop_last=False)

    return train_loader, val_loader


def get_split_val_loader(dataset_name, batch_size, num_workers, pred_path, label_path, img_h, img_w):
    try:
        supported_dataset = {
            "NHR": Dataset_OHAZE_val_split,
            "NHM": Dataset_OHAZE_val_split,
            "NHCL": Dataset_OHAZE_val_split,
            "NHCM": Dataset_OHAZE_val_split,
            "NHCD": Dataset_OHAZE_val_split,
            "UNREAL_NH": Dataset_OHAZE_val_split,
            "UNREAL_NH_NoSky_Dark": Dataset_OHAZE_val_split,
            "GTA5": Dataset_OHAZE_val_split,
            "NightHaze": Dataset_OHAZE_val_split,
            "YellowHaze": Dataset_OHAZE_val_split,
            "RWNHC_MM23_PseudoLabel": Dataset_OHAZE_val_split,
            "RWNHC_MM23_PseudoLabel_mini": Dataset_OHAZE_val_split,
        }
    except:
        raise ValueError("Dataset {} not supported".format(data))
    img_size = [img_h, img_w]
    val_dataset = supported_dataset[dataset_name](pred_path=pred_path, label_path=label_path, img_size=img_size)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True,
                                             drop_last=False)
    return val_loader