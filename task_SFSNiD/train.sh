# train_SFSNiD_supervised
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NHR/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset NHR --total_epoches 100 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NHM/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset NHM --total_epoches 200 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NHCL/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset NHCL --total_epoches 200 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NHCM/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset NHCM --total_epoches 200 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NHCD/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset NHCD --total_epoches 200 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/GTA5/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset GTA5 --total_epoches 200 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/UNREAL_NH/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset UNREAL_NH --total_epoches 100 --lr 0.0001 --device cuda:4 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NightHaze/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset NightHaze --total_epoches 100 --lr 0.0001 --device cuda:0 --num_res 3
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/YellowHaze/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset YellowHaze --total_epoches 100 --lr 0.0001 --device cuda:0 --num_res 3

# train_SFSNiD_semi_supervised
python train_SFSNiD_semi_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_semi_supervised/RWNHC_MM23_PseudoLabel_kappa130/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset RWNHC_MM23_PseudoLabel --total_epoches 20 --lr 0.0001 --device cuda:0 --num_res 3 --patch_size 16 --bri_ratio 100 --bri_weight 20 --kappa 130
