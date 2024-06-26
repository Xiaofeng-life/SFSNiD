import sys
sys.path.append("..")
import torch
import torch.nn as nn
from metric import cal_metric
import argparse
from PIL import Image
import os
import torch.utils.data as data
import torchvision.transforms as tt


class InfOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--data_root_val_UNREAL_NH", type=str, default="F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH/val/")
        self.parser.add_argument("--val_batch_size", type=int, default=1)
        self.parser.add_argument("--pth_path", type=str, default="../results/MyNightDehazing/SFSNiD/UNREAL_NH/models/last_SFSNiD_UNREAL_NH.pth")


    def parse(self):
        parser = self.parser.parse_args()
        return parser


def inference_one2one(net: nn.Module, pth_path, dataloader):
    """
    对单张输入有雾图像-单张输出无雾图像的模型进行推理
    :param net:
    :param pth_path:
    :param dataloader:
    :param save_dir:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1:define and reload model
    net = net.to(device)
    net.load_state_dict(torch.load(pth_path, map_location="cuda:0"))
    net.eval()

    num_samples = 0
    total_ssim = 0
    total_psnr = 0
    with torch.no_grad():
        for data in dataloader:
            hazy = data["hazy"].to(device)
            clear = data["gt"].to(device)

            pred = net(hazy)

            pred = torch.clamp(pred, min=0, max=1)

            num_samples += 1
            cur_psnr = cal_metric.cal_batch_psnr(pred=pred,
                                                       gt=clear)
            cur_ssim = cal_metric.cal_batch_ssim(pred=pred,
                                                       gt=clear)
            total_psnr += cur_psnr
            total_ssim += cur_ssim

            print("cur psnr: {}, cur ssim: {}".format(cur_psnr, cur_ssim))

        psnr = total_psnr / num_samples
        ssim = total_ssim / num_samples

        info = "Final metrics, SSIM: " + str(ssim) + ", PSNR: " + str(psnr) + "\n"
        print(info)



class Dataset_OHAZE(data.Dataset):
    """
    适用于成对图像的读取，有雾图片和无雾图片的名称一一对应，分别存储在hazy和clear两个文件夹中
    """
    def __init__(self, path):
        super(Dataset_OHAZE, self).__init__()

        self.haze_imgs_dir = os.listdir(os.path.join(path, "hazy/"))
        self.haze_imgs = [os.path.join(path, "hazy/", img) for img in self.haze_imgs_dir]

        self.clear_dir = os.path.join(path, "clear/")

        self.trans_hazy = tt.Compose([tt.ToTensor()])

        self.trans_gt = tt.Compose([tt.ToTensor()])

        self.split = "/"
        # if platform.system() == "Windows":
        #     self.split = "\\"

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



if __name__ == "__main__":
    config = InfOptions().parse()
    from methods.MyNightDehazing.SFSNiD import build_net


    val_dataset = Dataset_OHAZE(path=config.data_root_val_UNREAL_NH)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=config.val_batch_size, shuffle=False,
                                num_workers=0, pin_memory=True,
                                drop_last=False)

    network = build_net(num_res=3).cuda()

    inference_one2one(net=network,
                      pth_path=config.pth_path,
                      dataloader=val_loader)
