# THIS DATASET CLASS IS USED FOR LOADING OF PRE-AUGMENTED OR OTHER PARALLEL DATA

import os
from pathlib import Path

import numpy as np
import librosa

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import load_wav_torch_to_np
from audio_zen.utils import basename, find_parallel_data, sample_fixed_length_data_aligned
from sklearn.utils import shuffle


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_dir_list,
            sr,
            limit = 2000,
            target_task =None,
            sample_for_wave_u_net=False,
            sub_sample_length=1.0
    ):
        """
        My Validation Set

        VALIDATION_SET_1
        |-- clean
        `-- noisy
            |-- no_reverb
            `-- with_reverb
        """
        super(Dataset, self).__init__()
        noisy_files_list = []

        for dataset_dir in dataset_dir_list:
            dataset_dir = Path(dataset_dir).expanduser().absolute()
            noisy_files_list += librosa.util.find_files((Path(dataset_dir) / "noisy").absolute())

        self.length = min(len(noisy_files_list), limit)
        self.noisy_files_list = shuffle(noisy_files_list)[:limit]
        self.sr = sr

        self.target_task = target_task
        self.sample_for_wave_u_net = sample_for_wave_u_net
        #self.sample_length=sample_length
        self.sample_length = int(np.floor(sub_sample_length * self.sr)) # this is only used for wave_u_net

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        use the absolute path of the noisy speech to find the corresponding clean speech.

        Notes
            with_reverb and no_reverb dirs have same-named files.
            If we use `basename`, the problem will be raised (cover) in visualization.

        Returns:
            noisy: [waveform...], clean: [waveform...], type: [reverb|no_reverb] + name
        """
        noisy_file_path = self.noisy_files_list[item]
        noisy_filename, _ = basename(noisy_file_path)

        
        
        speech_type = "all"

        clean_file_path = find_parallel_data (noisy_file_path, "clean")

        noisy, _ = load_wav_torch_to_np(noisy_file_path, sr=self.sr)
        clean, _ = load_wav_torch_to_np(clean_file_path, sr=self.sr)

        if self.target_task in ["BGMI_INFERENCE"]:
            bgm_file_path = find_parallel_data (noisy_file_path, "bgm")
            bgm, _ = load_wav_torch_to_np(bgm_file_path, sr=self.sr)
            origin_testset = str(Path(noisy_file_path).parts[-3])

            noisy_v_bgm = np.array([noisy, bgm])

            return noisy_v_bgm, str(origin_testset + noisy_filename)

        if self.target_task in ["BGMI_INFERENCE_PLAIN"]:

            origin_testset = str(Path(noisy_file_path).parts[-3])

            return noisy, str(origin_testset + noisy_filename)


        if self.target_task in ["BGMI"]:
            bgm_file_path = find_parallel_data (noisy_file_path, "bgm")
            bgm, _ = load_wav_torch_to_np(bgm_file_path, sr=self.sr)
            return noisy, clean, bgm, noisy_filename, speech_type

        

        if self.sample_for_wave_u_net:
            #mixture, clean = sample_fixed_length_data_aligned(noisy, clean, len(noisy))    
            mixture, clean = sample_fixed_length_data_aligned(noisy, clean, self.sample_length)    
            return mixture.reshape(1, -1), clean.reshape(1, -1), noisy_filename, speech_type

        return noisy, clean, noisy_filename, speech_type




# x = Dataset(dataset_dir_list = ["/home/benedikt/thesis/TRAIN_EVALUATION_DATA/DENOISING_21_06_24/VALIDATION_SET"], sr = 16000, limit = 400, sample_for_wave_u_net = True)

# import tqdm
# for i in tqdm.tqdm(range (200)):
#     print (x[i][0].shape, x[i][1].shape, x[i][3])