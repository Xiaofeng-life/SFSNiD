# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import torch
from methods.MyNightDehazing import options_MyNightDehazing
import os
from dataset.dataloader_ImageDehazing import get_train_val_loader
import torchvision


def dynamic_gamma_clear(batch_clear, mean_val_clear):
    for i in range(batch_clear.size(0)):
        cur = batch_clear[i].unsqueeze(0)

        cur_mean = torch.mean(cur)
        gamma_you_want = torch.log(torch.FloatTensor([0.2]).to(cur.device)) / torch.log(cur_mean)
        # print("gamma is: ", gamma_you_want)

        cur = torch.pow(cur, gamma_you_want)

        while torch.mean(cur) > mean_val_clear:
            cur = torch.pow(cur, 1.1)

        batch_clear[i] = cur

    return batch_clear


def dynamic_gamma_hazy(batch_hazy, mean_val_hazy):
    for i in range(batch_hazy.size(0)):
        cur = batch_hazy[i].unsqueeze(0)

        cur_mean = torch.mean(cur)
        gamma_you_want = torch.log(torch.FloatTensor([0.2]).to(cur.device)) / torch.log(cur_mean)
        # print("gamma is: ", gamma_you_want)

        cur = torch.pow(cur, gamma_you_want)

        while torch.mean(cur) > mean_val_hazy:
            cur = torch.pow(cur, 1.1)

        batch_hazy[i] = cur

    return batch_hazy


def train():
    MEAN_VALUE_CLEAR = 0.2284
    MEAN_VALUE_HAZY = 0.2485

    for i, loader in enumerate([train_loader, val_loader]):
        for data in loader:
            image_haze_ori = data["hazy"]
            image_clear_ori = data["gt"]

            #
            print("mean clear before: ", torch.mean(image_clear_ori))
            image_clear = dynamic_gamma_clear(batch_clear=image_clear_ori.clone(), mean_val_clear=MEAN_VALUE_CLEAR)
            print("mean clear after: ", torch.mean(image_clear))

            #
            print("mean hazy before: ", torch.mean(image_haze_ori))
            image_haze = dynamic_gamma_hazy(batch_hazy=image_haze_ori.clone(), mean_val_hazy=MEAN_VALUE_HAZY)
            print("mean hazy after: ", torch.mean(image_haze))

            # torchvision.utils.save_image(torch.concatenate([image_haze, image_clear]),
            #                              os.path.join(res_dir + "train/cat/", data["name"][0]))
            path = None
            if i == 0:
                path = "train"
            else:
                path = "val"
            torchvision.utils.save_image(torch.concatenate([image_clear]),
                                         os.path.join(res_dir + path + "/clear/", data["name"][0]))

            torchvision.utils.save_image(torch.concatenate([image_haze]),
                                         os.path.join(res_dir + path + "/hazy/", data["name"][0]))


if __name__ == "__main__":
    # 1：参数定义
    config = options_MyNightDehazing.Options().parse()

    # 2：数据准备
    train_loader, val_loader = get_train_val_loader(dataset=config.dataset, img_h=config.img_h, img_w=config.img_w,
                                                    train_batch_size=config.train_batch_size,
                                                    num_workers=config.num_workers,
                                                    if_flip=False, if_crop=False, crop_h=256, crop_w=256)

    res_dir = config.results_dir
    train()
