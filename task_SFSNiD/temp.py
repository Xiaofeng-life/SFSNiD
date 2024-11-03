import torch


from methods.MyNightDehazing.SFSNiD import build_net
network = build_net(num_res=3)
network.load_state_dict(torch.load("../results/MyNightDehazing/different_kappa/RWNHC_MM23_PseudoLabel_BriRatio100_Gamma100/models/last_SFSNiD_RWNHC_MM23_PseudoLabel.pth",
                                   map_location="cuda:0"))