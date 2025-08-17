import os


def look(ori_path, txt):
    files = os.listdir(ori_path)

    with open(txt, "a+") as op:
        for f in files:
            if "png" in f:
                op.write(f + "\n")


if __name__ == "__main__":
    # look(ori_path="F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH_NoSky_Dark/train/hazy",
    #      txt="UNREAL_NH_No_Sky_Dark.txt")
    look(ori_path="F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/RWNHC_MM23_PseudoLabel_CVPR24/train/hazy",
         txt="UNREAL_NH_No_Sky_Dark.txt")