import os
import matplotlib.pyplot as plt
import json
import torch


class LossWriter():
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def add(self, loss_name: str, loss, i: int):
        with open(os.path.join(self.save_dir, loss_name + ".txt"), mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


def write_metrics(path, epoch, ssim, psnr):
    with open(path, mode='a') as f:
        info = str(epoch) + " " + str(ssim) + " " + str(psnr) + "\n"
        f.write(info)
        f.close()


def write_metrics_psnr_ssim_uiqm(path, epoch, psnr, ssim, uiqm):
    with open(path, mode='a') as f:
        info = str(epoch) + " " + str(psnr) + " " + str(ssim) + " " + str(uiqm) + "\n"
        f.write(info)
        f.close()


def plot_loss(txt_name, x_label, y_label, title, save_name, font_size=13, legend=None):
    all_i = []
    all_val = []
    with open(txt_name, "r") as f:
        all_lines = f.readlines()
        for line in all_lines:
            sp = line.split(" ")
            i = int(sp[0])
            val = float(sp[1])
            all_i.append(i)
            all_val.append(val)
    plt.figure(figsize=(6, 4))
    plt.plot(all_i, all_val)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    if legend:
        plt.legend(legend, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.savefig(save_name, dpi=200, bbox_inches="tight")


def plot_loss_ave(txt_name, x_label, y_label, title, save_name, font_size=13, legend=None, ave_num=1):

    all_i = []
    all_val = []
    with open(txt_name, "r") as f:
        all_lines = f.readlines()
        for line in all_lines:
            sp = line.split(" ")
            i = int(sp[0])
            val = float(sp[1])
            all_i.append(i)
            all_val.append(val)

    mean_i = []
    mean_val = []
    gap = ave_num
    for i in range(len(all_val) // gap):
        cur_sum = 0
        for k in range(0, gap):
            cur_sum += all_val[i * gap + k]

        mean_i.append(i)
        mean_val.append(cur_sum / gap)

    plt.figure(figsize=(6, 4))
    plt.plot(mean_i, mean_val)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    if legend:
        plt.legend(legend, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_all_losses(losses_path):
    loss_path_list = os.listdir(losses_path)

    for cur_file in loss_path_list:
        if cur_file.endswith(".txt"):
            plot_loss_ave(txt_name=os.path.join(losses_path, cur_file), x_label="iteration", y_label="losss",
                      title=cur_file, save_name=os.path.join(losses_path, cur_file[:-4] + ".png"), ave_num=10)


def save_config_as_json(save_path, config):
    with open(save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def save_best_model(cur_psnr, cur_ssim, best_psnr, best_ssim, save_dir, network, model_name, dataset_name):
    if cur_psnr > best_psnr:
        best_psnr = cur_psnr
        torch.save(network.state_dict(), os.path.join(save_dir, "models", "best_psnr_" + model_name + "_" + dataset_name + ".pth"))
        # print("save best psnr")
    if cur_ssim > best_ssim:
        best_ssim = cur_ssim
        torch.save(network.state_dict(),  os.path.join(save_dir, "models", "best_ssim_" + model_name + "_" + dataset_name + ".pth"))
        # print("save best ssim")

    torch.save(network.state_dict(), os.path.join(save_dir, "models", "last_" + model_name + "_" + dataset_name + ".pth"))

    return best_ssim, best_psnr


def save_cur_model(save_dir, network, model_name, dataset_name, epochs):

    torch.save(network.state_dict(), os.path.join(save_dir, "models", str(epochs) + "_" + model_name + "_" + dataset_name + ".pth"))

    return None


if __name__ == "__main__":
    plot_all_losses(losses_path="../results/simvp_bs1_seq4/utils/")
