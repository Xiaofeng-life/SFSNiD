# -*- coding: utf-8 -*-
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--beta1", type=float, default=0.9)
        self.parser.add_argument("--beta2", type=float, default=0.999)
        self.parser.add_argument("--total_epoches", type=int, default=20)

        self.parser.add_argument("--dataset", type=str)

        self.parser.add_argument('--num_workers', type=int, default=8)
        self.parser.add_argument("--img_w", type=int)
        self.parser.add_argument("--img_h", type=int)

        self.parser.add_argument("--lr", type=float, default=0.0002)
        self.parser.add_argument("--train_batch_size", type=int)
        self.parser.add_argument("--val_batch_size", type=int, default=1)
        self.parser.add_argument("--results_dir", type=str)

        self.parser.add_argument("--device", type=str, default="cuda")

        self.parser.add_argument("--step_size", type=str, default=10)
        self.parser.add_argument("--step_gamma", type=str, default=0.97)

        self.parser.add_argument("--num_res", type=int, default=12)

        self.parser.add_argument("--num_gamma", type=int, default=4)

        self.parser.add_argument("--patch_size", type=int)
        self.parser.add_argument("--bri_ratio", type=int)
        self.parser.add_argument("--bri_weight", type=int)

        self.parser.add_argument("--net_type", type=str)
        self.parser.add_argument("--stage_nums", type=int)
        self.parser.add_argument("--FFT_Loss", type=str)

        self.parser.add_argument("--mask_ratio", type=int)

        self.parser.add_argument("--gamma", type=int)

    def parse(self):
        parser = self.parser.parse_args()
        return parser


if __name__ == "__main__":
    parser = Options()
    parser = parser.parse()
    print(parser)