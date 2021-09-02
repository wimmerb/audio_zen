import importlib
import os
import time
from copy import deepcopy
from functools import reduce
from pathlib import Path
import numpy as np

import torch


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of l1 checkpoint."
    model_checkpoint = torch.load(os.path.abspath(os.path.expanduser(checkpoint_path)), map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # load tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["l1"]


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists(), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def check_nan(tensor, key=""):
    if torch.sum(torch.isnan(tensor)) > 0:
        print(f"Found NaN in {key}")


class ExecutionTime:
    """
    Count execution time.

    Examples:
        timer = ExecutionTime()
        ...
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_module(path: str, args: dict = None, initialize: bool = True):
    """
    Load module dynamically with "args".

    Args:
        path: module path in this project.
        args: parameters that passes to the Class or the Function in the module.
        initialize: initialize the Class or the Function with args.

    Examples:
        Config items are as follows：

            [model]
            path = "model.full_sub_net.FullSubNetModel"
            [model.args]
            n_frames = 32
            ...

        This function will:
            1. Load the "model.full_sub_net" module.
            2. Call "FullSubNetModel" Class (or Function) in "model.full_sub_net" module.
            3. If initialize is True:
                instantiate (or call) the Class (or the Function) and pass the parameters (in "[model.args]") to it.
    """
    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]

    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def print_tensor_info(tensor, flag="Tensor"):
    def floor_tensor(float_tensor):
        return int(float(float_tensor) * 1000) / 1000

    print(
        f"{flag}\n"
        f"\t"
        f"max: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, "
        f"mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")


def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def merge_config(*config_dicts):
    """
    Deep merge configuration dicts.

    Args:
        *config_dicts: any number of configuration dicts.

    Notes:
        1. The values of item in the later configuration dict(s) will update the ones in the former dict(s).
        2. The key in the later dict must be exist in the former dict. It means that the first dict must consists of all keys.

    Examples:
        a = [
            "a": 1,
            "b": 2,
            "c": {
                "d": 1
            }
        ]
        b = [
            "a": 2,
            "b": 2,
            "c": {
                "e": 1
            }
        ]
        c = merge_config(a, b)
        c = [
            "a": 2,
            "b": 2,
            "c": {
                "d": 1,
                "e": 1
            }
        ]

    Returns:
        New deep-copied configuration dict.
    """

    def merge(older_dict, newer_dict):
        for new_key in newer_dict:
            if new_key not in older_dict:
                # Checks items in custom config must be within common config
                raise KeyError(f"Key {new_key} is not exist in the common config.")

            if isinstance(older_dict[new_key], dict):
                older_dict[new_key] = merge(older_dict[new_key], newer_dict[new_key])
            else:
                older_dict[new_key] = deepcopy(newer_dict[new_key])

        return older_dict

    return reduce(merge, config_dicts[1:], deepcopy(config_dicts[0]))


def prepare_device(n_gpu: int, keep_reproducibility=False):
    """
    Choose to use CPU or GPU depend on the value of "n_gpu".

    Args:
        n_gpu(int): the number of GPUs used in the experiment. if n_gpu == 0, use CPU; if n_gpu >= 1, use GPU.
        keep_reproducibility (bool): if we need to consider the repeatability of experiment, set keep_reproducibility to True.

    See Also
        Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        # possibly at the cost of reduced performance
        if keep_reproducibility:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.benchmark = False  # ensures that CUDA selects the same convolution algorithm each time
            torch.set_deterministic(True)  # configures PyTorch only to use deterministic implementation
        else:
            # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda:0")

    return device


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext

def find_parallel_data(path, parallel_dir_name):
    path = Path(path)
    return path.parents[1] / parallel_dir_name / path.parts[-1]




voice_type_aliases = {
    "soprano": ["soprano", "sopran", "soprani", "sopranos", "soprans", "s", "sop"],
    "alto": ["alto", 
             "alt", 
             "altos", 
             "contralto", 
             "contralt", 
             "contralts", 
             "contralti", 
             "mezzo", 
             "a", 
             "c", 
             "mezzo-soprano", 
             "altus"
            ],
    "tenor": ["tenor", "tenors", "tenori", "ten", "t"],
    "bass": [
        "bass",
        "bas",
        "bajo",
        "baix",
        "baixos",
        "basses",
        "bariton",
        "baríton",
        "baríton",
        "baritone",
        "b",
        "nois", # Means 'men' in Catalan
        "salazar",# Orfeó Vigatà/Opera - 22 - Ària Pere+cor/voices
        'pere', # Orfeó Vigatà/Opera - 18 - Cor soldats/voices
    ],
    "accompaniment": ["piano", "violin", "cello", "celli", "viola", 
                      "oboe", "flauto", "violino", "violini", "fagotto", "orgel",
                      "continuo", "orgel", "timpani", 
                      "violone", "klaver", "organ", "organo", "fagotti", "violins", 
                      "violoncellos", "accompaniment", "tromba",
                     ],
}
alias_to_voice_type = {
    alias: voice_type
    for voice_type, aliases in voice_type_aliases.items()
    for alias in aliases
}



def get_voice_type(part_name: str):
    # Remove suffix such as part number (e.g. remove 'II' from 'Tenor II')
    name = part_name.replace('_', ' ').replace('-', ' ').replace('í', 'i').strip()
    name = name.split()[0]
    name_old = name
    name = name.rstrip(".1234567890")
    name = name.lower()
    voice_type = alias_to_voice_type.get(name, None)
    if (voice_type == None):
        for part in voice_type_aliases.keys():
            if part in name:
#                 print ("WARNING: part was not correctly mapped to voice type: ", 
#                        part_name, "was matched to", part, "by inclusion")
                voice_type = part
    if (voice_type == None):
        for part in voice_type_aliases.keys():
            if part in part_name:
#                 print ("WARNING: part was not correctly mapped to voice type: ", 
#                        part_name, "was matched to", part, "by inclusion")
#                 print ("INFO: get_voice_type did not manage to process this partname correctly:", part_name)
                voice_type = part
    return voice_type