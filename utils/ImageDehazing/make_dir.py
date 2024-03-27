import os


def make_train_dir(res_dir):
    if os.path.exists(res_dir):
        raise ValueError("res_dir already exists, avoid overwriting !!!!!!")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
        os.mkdir(os.path.join(res_dir, "cat_images"))
        os.mkdir(os.path.join(res_dir, "best_PSNR_images"))
        os.mkdir(os.path.join(res_dir, "best_SSIM_images"))
        os.mkdir(os.path.join(res_dir, "last_images"))
        os.mkdir(os.path.join(res_dir, "models"))
        os.mkdir(os.path.join(res_dir, "metrics"))
        os.mkdir(os.path.join(res_dir, "losses"))
        os.mkdir(os.path.join(res_dir, "configs"))
        os.mkdir(os.path.join(res_dir, "sample_images"))