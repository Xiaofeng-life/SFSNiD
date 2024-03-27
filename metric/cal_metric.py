from metric.ssim_function import ssim
from metric.psnr_function import psnr
import numpy as np


def tensor2img(ten):
    """
    Convert tensor to numpy
    :param ten: range [0, 1], format (C, H, W)
    :return: range [0, 255], (H, W, C)
    """
    ten = ten.clamp(min=0, max=1)
    ten = ten.detach().cpu() * 255.0
    ten = ten.numpy().transpose(1, 2, 0).astype(np.uint8)
    return ten


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cal_batch_psnr(pred, gt):

    assert pred.size() == gt.size()
    sum_psnr = 0
    for i in range(pred.size(0)):
        cur_psnr = psnr(pred[i], gt[i])
        if cur_psnr >= 100:
            raise ValueError("Maybe an error")
        else:
            sum_psnr += cur_psnr

    ave_psnr = sum_psnr / pred.size(0)
    return ave_psnr


def cal_batch_ssim(pred, gt):
    return ssim(img1=pred, img2=gt).cpu().numpy()
