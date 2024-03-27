import torch
from metric.cal_metric import cal_batch_psnr, cal_batch_ssim
from utils.ImageDehazing.writer import write_metrics
import os
import torchvision


def eval_SISO(val_loader, network, device, save_dir, if_eval_best, if_eval_last, network_name, dataset_name):
    descript = "_" + network_name + "_" + dataset_name

    def cell(val_loader, network, device, save_dir, pth_name, metric_name, save_type):
        network.load_state_dict(torch.load(os.path.join(save_dir, "models", pth_name)))
        ssim, psnr = eval_SISO_cell(val_loader=val_loader, network=network, device=device, save_dir=save_dir,
                                    if_save_cat=False, save_type=save_type)
        write_metrics(os.path.join(save_dir, "metrics", metric_name), epoch=0, ssim=ssim, psnr=psnr)

    if if_eval_best:
        cell(val_loader=val_loader, network=network, device=device, save_dir=save_dir,
             pth_name="best_psnr" + descript + ".pth",
             metric_name="best_PSNR" + descript + ".txt", save_type="best_PSNR_images")
        cell(val_loader=val_loader, network=network, device=device, save_dir=save_dir,
             pth_name="best_psnr" + descript + ".pth",
             metric_name="best_SSIM" + descript + ".txt", save_type="best_SSIM_images")

    if if_eval_last:
        cell(val_loader=val_loader, network=network, device=device, save_dir=save_dir,
             pth_name="last" + descript + ".pth",
             metric_name="last" + descript + ".txt", save_type="last_images")

    return None


def eval_SISO_cell(val_loader, network, device, save_dir, if_save_cat, save_type):
    network.eval()
    with torch.no_grad():
        num_samples = 0
        total_ssim = 0
        total_psnr = 0
        for data in val_loader:
            image_haze = data["hazy"].to(device)
            image_clear = data["gt"].to(device)
            img_name = data["name"]

            out = network(image_haze)
            out = torch.clamp(out, min=0, max=1)

            num_samples += 1
            total_psnr += cal_batch_psnr(pred=out, gt=image_clear)
            total_ssim += cal_batch_ssim(pred=out, gt=image_clear)

            for j in range(out.size(0)):
                if if_save_cat:
                    out_cat = torch.cat((image_haze, out, image_clear), dim=3)
                else:
                    out_cat = out
                torchvision.utils.save_image(out_cat[j].unsqueeze(0),
                                             os.path.join(save_dir, save_type, img_name[j]))

        psnr = total_psnr / num_samples
        ssim = total_ssim / num_samples

    return ssim, psnr
