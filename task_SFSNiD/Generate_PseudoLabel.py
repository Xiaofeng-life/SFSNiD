import sys
sys.path.append("..")
import torch
import os
import argparse
from PIL import Image
import numpy as np


class InfOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset", type=str)
        self.parser.add_argument("--img_w", type=int)
        self.parser.add_argument("--img_h", type=int)
        self.parser.add_argument("--val_batch_size", type=int, default=1)
        self.parser.add_argument("--results_dir", type=str)
        self.parser.add_argument("--net", type=str)
        self.parser.add_argument("--pth_path", type=str)
        self.parser.add_argument("--num_workers", type=int, default=4)

    def parse(self):
        parser = self.parser.parse_args()
        return parser


def inference_no_gt(data_root, save_path, net,
                    pth_path,
                    img_size,
                    clip_min=0, clip_max=1):
    file_list = os.listdir(data_root)
    net = net.cuda()
    net.load_state_dict(torch.load(pth_path, map_location="cuda:0"))
    net.eval()
    with torch.no_grad():
        for file in file_list:
            img_path = os.path.join(data_root, file)
            img = Image.open(img_path)
            hazy_shape = (img.width, img.height)

            # pre-process
            img = img.resize(img_size)
            img = np.array(img).astype(np.float32)
            img = torch.from_numpy(img) / 255.0
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0).cuda()

            # predict
            pred = net(img)
            pred = torch.clamp(pred, min=clip_min, max=clip_max)

            # cat
            # pred = torch.concatenate([img, pred], dim=3)

            pred = pred[0].detach().cpu().numpy().transpose(1, 2, 0)
            pred = (pred * 255).astype(np.uint8)
            pred = Image.fromarray(pred)
            pred = pred.resize(hazy_shape)
            print("save to :{}".format(os.path.join(save_path, file)))
            pred.save(os.path.join(save_path, file))


def get_network(network_name):
    if network_name == "SFSNiD":
        from methods.MyNightDehazing.SFSNiD import build_net
        network =build_net(num_res=3)

    else:
        raise ValueError("model {} not supported!".format(network_name))
    
    return network


if __name__ == "__main__":
    config = InfOptions().parse()

    path_dict = {
                 "RWNHC_MM23": "RWNHC_MM23/all_hazy/",
    }

    data_root_val = path_dict[config.dataset]

    results_dir = config.results_dir
    os.mkdir(results_dir)
    # results_dir = os.path.join(results_dir, "images")
    # os.mkdir(results_dir)

    network = get_network(network_name=config.net)

    img_size = [config.img_h, config.img_w]
    inference_no_gt(data_root=data_root_val,
                    save_path=results_dir,
                    net=network,
                    pth_path=config.pth_path,
                    img_size=img_size)