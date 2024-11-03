import os


def look(ori_path, txt):
    files = os.listdir(ori_path)

    with open(txt, "a+") as op:
        for f in files:
            op.write(f + "\n")


if __name__ == "__main__":
    look(ori_path="F:/CXF_Code/dataset/processed_dataset/night_dehazing_dataset/UNREAL_NH_NoSky_Dark/train/hazy",
         txt="UNREAL-NH-D.txt")