server = "3090"


def get_path_dict_ImageDehazing():

    if server == "PC":
        path_dict = {
            "NHM":
                {
                    "train": "F:/dataset/processed_dataset/night_dehazing_dataset/3R/NHM/train/",
                    "val": "F:/dataset/processed_dataset/night_dehazing_dataset/3R/NHM/val/"
                },

        }

    elif server == "3090":
        path_dict = {

            "NHR":
                {
                    "train": "../../night_dehazing_dataset/3R/NHR/train/",
                    "val": "../../night_dehazing_dataset/3R/NHR/val/"
                },

            "NHM":
                {
                    "train": "../../night_dehazing_dataset/3R/NHM/train/",
                    "val": "../../night_dehazing_dataset/3R/NHM/val/"
                },

            "NHCL":
                {
                    "train": "../../night_dehazing_dataset/3R/NHCL/train/",
                    "val": "../../night_dehazing_dataset/3R/NHCL/val/"
                },

            "NHCM":
                {
                    "train": "../../night_dehazing_dataset/3R/NHCM/train/",
                    "val": "../../night_dehazing_dataset/3R/NHCM/val/"
                },

            "NHCD":
                {
                    "train": "../../night_dehazing_dataset/3R/NHCD/train/",
                    "val": "../../night_dehazing_dataset/3R/NHCD/val/"
                },

            "UNREAL_NH":
                {
                    "train": "../../night_dehazing_dataset/UNREAL_NH/train/",
                    "val": "../../night_dehazing_dataset/UNREAL_NH/val/"
                },

            "UNREAL_NH_NoSky_Dark":
                {
                    "train": "../../night_dehazing_dataset/UNREAL_NH_NoSky_Dark/train/",
                    "val": "../../night_dehazing_dataset/UNREAL_NH_NoSky_Dark/val/"
                },

            "GTA5":
                {
                    "train": "../../night_dehazing_dataset/GTA5/train/",
                    "val": "../../night_dehazing_dataset/GTA5/val/"
                },

            "NightHaze":
                {
                    "train": "../../night_dehazing_dataset/HDP_dataset/NightHaze/train/",
                    "val": "../../night_dehazing_dataset/HDP_dataset/NightHaze/val/"
                },

            "YellowHaze":
                {
                    "train": "../../night_dehazing_dataset/HDP_dataset/YellowHaze/train/",
                    "val": "../../night_dehazing_dataset/HDP_dataset/YellowHaze/val/"
                },

            "RWNHC_MM23_PseudoLabel":
                {
                    "train": "../../night_dehazing_dataset/RWNHC_MM23_PseudoLabel/train/",
                    "val": "../../night_dehazing_dataset/RWNHC_MM23_PseudoLabel/val/"
                },
        }

    else:
        path_dict = None
    return path_dict
