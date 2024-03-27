# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import torch.optim as optim
import torch.nn as nn
import torch
from methods.MyNightDehazing.SFSNiD import build_net
from methods.MyNightDehazing import options_MyNightDehazing
import os
from utils.ImageDehazing.writer import LossWriter, plot_all_losses, write_metrics, save_config_as_json, save_best_model
from dataset.dataloader_ImageDehazing import get_train_val_loader
from utils.ImageDehazing.make_dir import make_train_dir
from methods.ImageDehazing.epoch_eval import eval_SISO, eval_SISO_cell
import torch.nn.functional as F


def train():
    # 5：
    iteration = 0
    best_psnr = 0
    best_ssim = 0
    for epoch in range(config.total_epoches):
        network.train()
        for data in train_loader:
            image_haze = data["hazy"].to(device)
            image_clear = data["gt"].to(device)

            # #################################################
            optimizer.zero_grad()
            generated_image = network(image_haze)
            image_clear2 = F.interpolate(image_clear, scale_factor=0.5, mode='bilinear')
            image_clear4 = F.interpolate(image_clear, scale_factor=0.25, mode='bilinear')
            l1 = loss_func(generated_image[0], image_clear4)
            l2 = loss_func(generated_image[1], image_clear2)
            l3 = loss_func(generated_image[2], image_clear)
            loss_content = l1 + l2 + l3

            label_fft1 = torch.fft.fft2(image_clear4, dim=(-2, -1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(generated_image[0], dim=(-2, -1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(image_clear2, dim=(-2, -1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(generated_image[1], dim=(-2, -1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(image_clear, dim=(-2, -1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(generated_image[2], dim=(-2, -1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = loss_func(pred_fft1, label_fft1)
            f2 = loss_func(pred_fft2, label_fft2)
            f3 = loss_func(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3

            loss = loss_content + 0.1 * loss_fft

            loss.backward()
            optimizer.step()

            loss_writer.add("loss", loss.item(), iteration)


            iteration += 1

            if iteration % 100 == 0:
                print("Iter {}, Loss is {}".format(iteration, loss.item()))

            # #################################################
        scheduler.step()
        cur_lr = optimizer.param_groups[-1]['lr']
        print("current lr is: {}".format(cur_lr))

        network.eval()
        ssim, psnr = eval_SISO_cell(val_loader=val_loader, network=network, device=device, save_dir=res_dir,
                                    if_save_cat=True, save_type="cat_images")
        write_metrics(os.path.join(res_dir, "metrics/metric.txt"), epoch=epoch, ssim=ssim, psnr=psnr)
        print("SFSNiD: ||iterations: {}||, ||PSNR {:.4}||, ||SSIM {:.4}||".format(iteration, psnr, ssim))

        best_ssim, best_psnr = save_best_model(cur_psnr=psnr, cur_ssim=ssim, best_psnr=best_psnr,
                                               best_ssim=best_ssim, save_dir=res_dir, network=network,
                                               model_name="SFSNiD", dataset_name=config.dataset)

        plot_all_losses(losses_path=os.path.join(res_dir, "losses"))

    # ################################################################################## #
    # eval
    eval_SISO(val_loader=val_loader, network=network, device=device, save_dir=res_dir,
              if_eval_best=True, if_eval_last=True, network_name="SFSNiD", dataset_name=config.dataset)


if __name__ == "__main__":
    config = options_MyNightDehazing.Options().parse()
    device = torch.device(config.device)
    train_loader, val_loader = get_train_val_loader(dataset=config.dataset, img_h=config.img_h, img_w=config.img_w,
                                                    train_batch_size=config.train_batch_size,
                                                    num_workers=config.num_workers,
                                                    if_flip=True, if_crop=False, crop_h=256, crop_w=256)


    network = build_net(num_res=config.num_res).to(device)
    res_dir = config.results_dir

    if os.path.exists(res_dir):
        network.load_state_dict(torch.load(os.path.join(res_dir, "models", "last_SFSNiD_" + config.dataset + ".pth"),
                                           map_location=config.device))
    else:
        make_train_dir(res_dir)

    loss_writer = LossWriter(os.path.join(res_dir, "losses"))
    save_config_as_json(save_path=os.path.join(res_dir, "configs", "config.txt"), config=config)

    optimizer = optim.Adam(network.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.step_size, gamma=config.step_gamma)
    loss_func = nn.L1Loss()

    train()
