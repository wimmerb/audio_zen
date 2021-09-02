# THIS DATASET CLASS IS USED FOR DYNAMIC AUGMENTATION DURING TRAINING

import random
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm

import sys
sys.path.append("/home/benedikt/thesis/repos/FullSubNet")
from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, save_wav, subsample, subsample_audio_tensor
from audio_zen.utils import expand_path, basename, sample_fixed_length_data_aligned

from itertools import product
import torch
import torchaudio
from sklearn.utils import shuffle
import traceback


class Dataset(BaseDataset):
    
    def __init__(self,
                 clean_dataset,
                 clean_dataset_limit,
                 clean_dataset_offset,
                 noise_dataset,
                 noise_dataset_limit,
                 noise_dataset_offset,
                 rir_dataset,
                 rir_dataset_limit,
                 rir_dataset_offset,
                 snr_range,
                 reverb_proportion,
                 pass_proportion,
                 pitch_shift_proportion,
                 pitch_shift_range,
                 silent_target_proportion,
                 silence_length,
                 target_dB_FS,
                 target_dB_FS_floating_value,
                 sub_sample_length,
                 sr,
                 pre_load_clean_dataset,
                 pre_load_noise,
                 pre_load_rir,
                 num_workers,
                 buffer_size,
                 buffer_use_only_identity,
                 target_task,
                 sample_for_wave_u_net=False,
                 sample_length=None,
                 debug_dump_limit=0
                 ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()

        # acoustics args
        self.sr = sr
        print ("init", num_workers)

        # parallel args
        self.num_workers = num_workers

        clean_dataset_list = [line.rstrip('\n') for line in open(expand_path(clean_dataset), "r")]
        noise_dataset_list = [line.rstrip('\n') for line in open(expand_path(noise_dataset), "r")]
        rir_dataset_list = [line.rstrip('\n') for line in open(expand_path(rir_dataset), "r")]

        clean_dataset_list = self._offset_and_limit(clean_dataset_list, clean_dataset_offset, clean_dataset_limit)
        noise_dataset_list = self._offset_and_limit(noise_dataset_list, noise_dataset_offset, noise_dataset_limit)
        rir_dataset_list = self._offset_and_limit(rir_dataset_list, rir_dataset_offset, rir_dataset_limit)

        if pre_load_clean_dataset:
            clean_dataset_list = self._preload_dataset(clean_dataset_list, remark="Clean Dataset")

        if pre_load_noise:
            noise_dataset_list = self._preload_dataset(noise_dataset_list, remark="Noise Dataset")

        if pre_load_rir:
            rir_dataset_list = self._preload_dataset(rir_dataset_list, remark="RIR Dataset")

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list
        self.rir_dataset_list = rir_dataset_list

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        assert 0 <= reverb_proportion <= 1, "proportion should be in [0, 1]"
        self.reverb_proportion = reverb_proportion
        assert 0 <= pass_proportion <= 1, "proportion should be in [0, 1]"
        self.pass_proportion = pass_proportion
        assert 0 <= pitch_shift_proportion <= 1, "proportion should be in [0, 1]"
        self.pitch_shift_proportion = pitch_shift_proportion
        pitch_shift_list = self._parse_snr_range(pitch_shift_range)
        self.pitch_shift_list = pitch_shift_list
        assert 0 <= silent_target_proportion <= 1, "proportion should be in [0, 1]"
        self.silent_target_proportion = silent_target_proportion

        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        #self.sub_sample_length = sub_sample_length

        self.length = len(self.clean_dataset_list)

        self.buffer_size = buffer_size # PRODUCES buffersize**2 entries (see buffer_positions)
        self.buffer_offset = 0
        self.buffer_picking_item_position = 0
        self.buffer_needs_redo = True
        self.buffer_clean_tensors = []
        self.buffer_picking_order = []
        self.buffer_convoluted_speech = None
        self.buffer_noise = []
        self.buffer_use_only_identity = buffer_use_only_identity

        #TODO check this assertion
        assert target_task in ['denoise', 'dereverb', 'denoise+dereverb'], f"target_task not in {['denoise', 'dereverb', 'denoise+dereverb']}"
        self.target_task = target_task

        self.sample_for_wave_u_net = sample_for_wave_u_net
        #self.sample_length=sample_length
        self.sample_length = int(np.floor(sub_sample_length * self.sr))
        if sample_length != None:
            print ("Dynamic Dataset: Sample length derived from sample_length parameter, IGNORING SUB_SAMPLE_LENGTH")
            self.sample_length = sample_length

        self.curr_filename = ""

        self.buffer_clean_files_names = []
        self.debug_dump_limit = debug_dump_limit

    def __len__(self):
        return self.length

    def _append_current_filename (self, text = ""):
        self.curr_filename += text
    
    def pop_current_filename(self):
        tmp = self.curr_filename+".wav"
        self.curr_filename = ""
        return tmp
    
    

    def _preload_dataset(self, file_path_list, remark=""):
        waveform_list = Parallel(n_jobs=self.num_workers)(
            delayed(load_wav)(f_path, self.sr) for f_path in tqdm(file_path_list, desc=remark)
        )
        return list(zip(file_path_list, waveform_list))

    #@staticmethod
    def _random_select_from(self, dataset_list):
        return random.choice(dataset_list)


    def _select_noise_y(self, target_length):
        assert "denoise" in self.target_task, "_select_noise_y accessed despite inexistent denoising target"

        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        # TODO weave into complete 'filename'
        noise_fns = "_NOISE"

        pitch_shift = 0
        if bool(np.random.random(1) < self.pitch_shift_proportion):
            pitch_shift = self._random_select_from (self.pitch_shift_list)
        

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)

            noise_fns += "_" + basename(noise_file)[0]

            noise_new_added = (self._get_sample_norm(noise_file, resample = self.sr, processed = True, pitch_shift = pitch_shift)[0])
            noise_new_added = np.transpose (np.array (noise_new_added))

            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # Adding silence between snippets of noise
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        # WE HANDLE THIS IN POP_ITEM NOW
        # if len(noise_y) > target_length:
        #     idx_start = np.random.randint(len(noise_y) - target_length)
        #     noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y, noise_fns

    # @staticmethod
    def snr_mix(self, clean_y, noise_y, conv_y, snr, target_dB_FS, target_dB_FS_floating_value, eps=1e-6, sr=16000, reverb_proportion=0,
            pass_proportion=0,
            silent_target_proportion=0):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        use_reverb = bool(np.random.random(1) < reverb_proportion)
        do_pass = bool(np.random.random(1) < pass_proportion)
        do_silent_target = bool(np.random.random(1) < silent_target_proportion) and ("denoise" in self.target_task)
        
        if do_silent_target:
            return noise_y, np.zeros_like(clean_y)

        if do_pass:
            return clean_y, clean_y

        if self.target_task == "dereverb":
            # clean_y is target, clean_y_reverberant is mixture
            if use_reverb:
                clean_y_reverberant = conv_y
            else:
                clean_y_reverberant = clean_y
            clean_y = clean_y
        elif self.target_task == "denoise":
            # clean_y is target (can be reverberant), mixture will be created from clean_y
            if use_reverb:
                clean_y = conv_y
            else:
                clean_y = clean_y
            clean_y_reverberant = None
        elif self.target_task == "denoise+dereverb":
            # clean_y is target, clean_y_reverberant will be combined with noise to give mixture
            clean_y = clean_y
            if use_reverb:
                clean_y_reverberant = conv_y
            else:
                clean_y_reverberant = clean_y

        clean_y, _ = norm_amplitude(clean_y, soft_fail = True)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        if not clean_y_reverberant is None: # we are in a dereverb or denoise+dereverb scenario
            clean_y_reverberant, _ = norm_amplitude(clean_y_reverberant, soft_fail = True)
            clean_y_reverberant, _, _ = tailor_dB_FS(clean_y_reverberant, target_dB_FS)
            clean_reverberant_rms = (clean_y_reverberant ** 2).mean() ** 0.5

            #for debugging
            #assert np.abs (clean_rms-clean_reverberant_rms) < 100*eps, f"rms of clean_rms and clean_reverberant_rms is unequal{clean_rms} vs {clean_reverberant_rms}, distance is  {np.abs (clean_rms-clean_reverberant_rms)}"

            mixture = clean_y_reverberant
            target = clean_y

        else: #denoising scenario
            mixture = clean_y
            target = clean_y

        if "denoise" in self.target_task:
            noise_y, _ = norm_amplitude(noise_y, soft_fail = True)
            noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
            noise_rms = (noise_y ** 2).mean() ** 0.5

            snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
            noise_y *= snr_scalar

            mixture = mixture + noise_y
        
        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        # scale both target and mixture amplitude according to random mixture target rms
        mixture, _, noisy_scalar = tailor_dB_FS(mixture, noisy_target_dB_FS)
        target *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(mixture):
            noisy_y_scalar = np.max(np.abs(mixture)) / (0.99 - eps)  # 相当于除以 1
            mixture = mixture / noisy_y_scalar
            target = target / noisy_y_scalar

        return mixture, target
    
    # # @staticmethod
    # def snr_mix(self, clean_y, noise_y, conv_y, snr, target_dB_FS, target_dB_FS_floating_value, eps=1e-6, sr=16000, reverb_proportion=0,
    #         pass_proportion=0,
    #         silent_target_proportion=0):
    #     """
    #     混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

    #     Args:
    #         clean_y: 纯净语音
    #         noise_y: 噪声
    #         snr (int): 信噪比
    #         target_dB_FS (int):
    #         target_dB_FS_floating_value (int):
    #         rir: room impulse response, None 或 np.array
    #         eps: eps

    #     Returns:
    #         (noisy_y，clean_y)
    #     """
    #     use_reverb = bool(np.random.random(1) < reverb_proportion)
    #     do_pass = bool(np.random.random(1) < pass_proportion)
    #     do_silent_target = bool(np.random.random(1) < silent_target_proportion)

    #     if do_silent_target:
    #         return noise_y, np.zeros_like(clean_y)

    #     if do_pass:
    #         return clean_y, clean_y
        
    #     if use_reverb:
    #         clean_y = conv_y
    #     else:
    #         clean_y = clean_y
        
    #     #TODO implement dereverb,denoise combinations

    #     clean_y, _ = norm_amplitude(clean_y)
    #     clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
    #     clean_rms = (clean_y ** 2).mean() ** 0.5

    #     # clean_y_reverberant, _ = norm_amplitude(clean_y_reverberant)
    #     # clean_y_reverberant, _, _ = tailor_dB_FS(clean_y_reverberant, target_dB_FS)
    #     # clean_reverberant_rms = (clean_y_reverberant ** 2).mean() ** 0.5

    #     # if(rir is not None):
    #     #     print (clean_rms, clean_reverberant_rms)

    #     noise_y, _ = norm_amplitude(noise_y)
    #     noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
    #     noise_rms = (noise_y ** 2).mean() ** 0.5

    #     snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
    #     noise_y *= snr_scalar


    #     noisy_y = clean_y + noise_y
        
    #     # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
    #     noisy_target_dB_FS = np.random.randint(
    #         target_dB_FS - target_dB_FS_floating_value,
    #         target_dB_FS + target_dB_FS_floating_value
    #     )

    #     # 使用 noisy 的 rms 放缩音频
    #     noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
    #     clean_y *= noisy_scalar

    #     # 合成带噪语音的时候可能会 clipping，虽然极少
    #     # 对 noisy, clean_y, noise_y 稍微进行调整
    #     if is_clipped(noisy_y):
    #         noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
    #         noisy_y = noisy_y / noisy_y_scalar
    #         clean_y = clean_y / noisy_y_scalar

    #     return noisy_y, clean_y

    #@staticmethod
    @classmethod
    def _get_sample(cls, path, resample=None, pitch_shift = 0):
        effects = [
            ["remix", "1"]
        ]

        if pitch_shift != 0:
            effects.append(["pitch", str(int (pitch_shift))])

        if resample:
            effects.append(["rate", f'{resample}'])

        return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

    
    
    #@staticmethod
    @classmethod
    def _get_sample_norm(cls, snd_path, resample=None, processed=False, reverted=False, pitch_shift = 0):
        # print ("RIR SAMPLE")
        snd_raw, sample_rate = cls._get_sample(snd_path, resample=resample, pitch_shift = pitch_shift)
        # print (rir_raw.shape)
        if not processed:
            return snd_raw, sample_rate
        # print(rir_raw.shape)
        snd = snd_raw # [:, int(sample_rate*1.01):int(sample_rate*1.3)]
        snd = snd / torch.norm(snd, p=2)
        if reverted:
            snd = torch.flip(snd, [1])
        # print (rir.shape)
        return snd, sample_rate


    def fill_conv_buffer(self):
        

        clean_files = []
        for index in range(self.buffer_offset, self.buffer_offset + self.buffer_size):
            index = index % self.length
            clean_files.append( self.clean_dataset_list[index])
        #print (clean_files)
        self.buffer_clean_files_names = clean_files
        
        self.buffer_offset = (self.buffer_offset + self.buffer_size) % self.length
        #print ("BUFFER OFFSET", self.buffer_offset)
        

        self.buffer_clean_tensors = []
        for fn in clean_files:
            #clean_y = load_wav(fn, sr=self.sr)
            clean_y, sample_rate = self._get_sample_norm(fn, resample=self.sr)
            clean_y = subsample_audio_tensor (clean_y, sub_sample_length=self.sample_length)
            # print (clean_y.shape)
            self.buffer_clean_tensors.append (clean_y)

        rirs = []
        for i in range (self.buffer_size):
            rir_file = self._random_select_from(self.rir_dataset_list)
            rirs.append (self._get_sample_norm(rir_file, resample = self.sr, processed = True, reverted = True)[0])
            #rirs.append (load_wav(rir_file, sr=self.sr))

        
        # print ("DONE LOADING")
        # print ([x.shape for x in self.buffer_clean_tensors])
        # print ("XXXXXXX")
        # print ([x.shape for x in rirs])
        rirs = np.array(rirs, dtype=object)

        self.rirs = rirs

        # print ("XXXXXXX")

        with torch.no_grad():
            max_rir_len = max([rir_sample.shape[1] for rir_sample in rirs])
            speech_ = torch.stack([torch.nn.functional.pad(clean_y, (max_rir_len-1,0)) for clean_y in self.buffer_clean_tensors.copy()]).cuda()
            rir_ = torch.stack([torch.nn.functional.pad(rir_sample, (max_rir_len-rir_sample.shape[1],0)) for rir_sample in rirs]).cuda()

            # print (speech_.device)

            # print (speech_.shape)
            # print (rir_.shape)

            speech = torch.nn.functional.conv1d(speech_, rir_)

            #print ("out...", speech.device)

            #print (self.sub_sample_length * self.sr)
            #print (speech.shape)

            self.buffer_convoluted_speech = speech.cpu()
            
            del speech
            del speech_
            del rir_
            # conv_1 = np.array(speech[1,1,:])
            # conv_2 = np.array(speech[1,2,:])

            # save_wav("conv_1.wav", conv_1, self.sr)
            # save_wav("conv_2.wav", conv_2, self.sr)

            #print ("BUFFERING NOISE...")
            if ("denoise" in self.target_task):
                self.buffer_noise = [self._select_noise_y(target_length = self.sample_length) for _ in range(self.buffer_size)]
            else:
                self.buffer_noise = None

    def pop_from_buffer(self):
        

        if self.buffer_needs_redo:
            
            self.fill_conv_buffer()
            self.buffer_picking_item_position = 0
            if self.buffer_use_only_identity:
                self.buffer_picking_order = shuffle([(x,x) for x in range(self.buffer_size)])
                #TODO CHECK FUNCTIONALITY
            else:
                self.buffer_picking_order = shuffle([x for x in product(range(self.buffer_size), repeat=2)])
            self.buffer_needs_redo = False

            #print ("NEW BUFFER")
            #print (self.buffer_offset)
            #print ("SAMPLE NR:", self.buffer_offset*self.buffer_size)
        
        speech_pos, rir_pos = self.buffer_picking_order[self.buffer_picking_item_position]

        clean_y = np.transpose (np.array (self.buffer_clean_tensors[speech_pos]))[:,0]
        #print ("CLEAN SHAPE", clean_y.shape)
        conv_y = np.transpose (np.array (self.buffer_convoluted_speech [speech_pos, rir_pos, :]))
        #print ("CONV SHAPE", conv_y.shape)
        assert clean_y.shape[0] == conv_y.shape[0]

        if "denoise" in self.target_task:
            if self.buffer_use_only_identity:
                noise_y, noise_names = self.buffer_noise[self.buffer_picking_item_position]
            else:
                noise_y, noise_names = random.choice(self.buffer_noise)
        else:
            noise_y, noise_names = (None, "")
        
        self.buffer_picking_item_position += 1
        if self.buffer_picking_item_position >= len(self.buffer_picking_order):
            self.buffer_needs_redo = True
        
        # print ("---------------------")
        # print (len (self.buffer_noise))
        # print (self.buffer_size)
        # print (len (self.buffer_picking_order))
        # print ("---------------------")
        

        self._append_current_filename (basename (self.buffer_clean_files_names[speech_pos])[0])
        assert self.buffer_clean_files_names[speech_pos] == self.clean_dataset_list[self.buffer_offset+speech_pos-self.buffer_size] #just for debugging
        self._append_current_filename (noise_names)
        
        if "denoise" in self.target_task:
            target_length = self.sample_length
            if len(noise_y) > target_length: # TODO needed?
                idx_start = np.random.randint(len(noise_y) - target_length)
                noise_y = noise_y[idx_start:idx_start + target_length]

                assert len(clean_y) == len(noise_y) == self.sample_length, f"Inequality in length: noise vs clean: {len(clean_y)} {len(noise_y)}, desired:{self.sample_length}"
        else:
            assert noise_y == None, "noise should be none as we do not target denoising"
            assert self.buffer_noise == None, "buffer_noise sould be none as we do not taret denoising"

        # save_wav(f"{speech_pos}_{rir_pos}_clean.wav", clean_y, self.sr)
        # save_wav(f"{speech_pos}_{rir_pos}_conv.wav", conv_y, self.sr)

        return clean_y, conv_y, noise_y

    def __getitem__(self, item):
        #get random item from buffer
        clean_y, conv_y, noise_y = self.pop_from_buffer()
        
        

        snr = self._random_select_from(self.snr_list)
        

        
        # rir = None
        # if use_reverb:
        #     rir_file = self._random_select_from(self.rir_dataset_list)
        #     rir = load_wav(rir_file, sr=self.sr)
        
        # # rir = load_wav(self._random_select_from(self.rir_dataset_list), sr=self.sr) if use_reverb else None

        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            conv_y=conv_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            sr=self.sr,
            reverb_proportion=self.reverb_proportion,
            pass_proportion=self.pass_proportion,
            silent_target_proportion=self.silent_target_proportion
        )


        if self.debug_dump_limit > 0:
            print ("writing files, task is:", self.target_task)
            self.debug_dump_limit -= 1
            if not os.path.exists ("AUGMENTATION_DEBUG_DUMP"):
                os.makedirs ("AUGMENTATION_DEBUG_DUMP")
            save_wav(Path ("AUGMENTATION_DEBUG_DUMP") / Path (f"{self.target_task}_{str(self.debug_dump_limit).zfill(3)}_clean.wav"), clean_y, self.sr)
            save_wav(Path ("AUGMENTATION_DEBUG_DUMP") / Path (f"{self.target_task}_{str(self.debug_dump_limit).zfill(3)}_noisy.wav"), noisy_y, self.sr)

        # noisy_y = noisy_y.astype(np.float32)
        # clean_y = clean_y.astype(np.float32)

        if self.sample_for_wave_u_net:
            mixture, clean = sample_fixed_length_data_aligned(noisy_y, clean_y, self.sample_length)    
            return mixture.reshape(1, -1), clean.reshape(1, -1)
            

        # print (item) #for debugging
        
        return noisy_y, clean_y

# import toml
# from audio_zen.utils import initialize_module
# if __name__ == '__main__':
#     path = "/home/benedikt/thesis/datasets/VCTK/wav48_silence_trimmed/p262/p262_388_mic1.wav"

#     print (Dataset._get_sample_norm(path, processed = True))
#     print (Dataset._get_sample_norm(path))

#     config = toml.load("creator_setting.toml")
#     dynamic_dataset=initialize_module(
#         config["dataset"]["path"],
#         args=config["dataset"]["args"])
#     bla = dynamic_dataset [0]

#     for i, rir in enumerate (dynamic_dataset.rirs):
#         print (rir)
#         save_wav(f"rir_{i}.wav", np.array(rir)[0], 16000)
