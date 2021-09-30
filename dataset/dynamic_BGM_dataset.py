from pathlib import Path
import librosa
import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append("/home/benedikt/thesis/repos/FullSubNet")

from audio_zen.acoustics.feature import load_wav_torch_to_np, load_wav_torch, subsample_audio_tensor, norm_amplitude, is_clipped, tailor_dB_FS
from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import get_voice_type

import tqdm
import time
import torch

from pprint import pprint
import warnings


class Dataset(BaseDataset):
    def __init__(self,
                file_list_fn,
                rir_list_fn,
                mir_list_fn,
                nr_aug_variations,
                mir_apply_cnt_list,
                rir_apply_cnt_list,
                sr,
                sub_sample_length,
                max_audio_length,
                aligned_prob,
                include_lead_in_accomp_prob,
                sample_for_validation_or_test_creation,
                nr_samples_per_voice_v_augmentation_combination,
                snr_range,
                target_dB_FS,
                target_dB_FS_floating_value,

                #params for get_random_voiced_part
                grvp_max_n_tries_foreground,
                grvp_max_n_tries_background,

                dry_run=False,
                eps=1e-6,
                target_task="standard",

                silent_lead_prob = 0.05,
                
                
                DEBUG_SINGULAR_BUFFER=False,
    ):
        super().__init__()

        print ("initializing dynamic BackGroundMusic dataset...")

        with open (file_list_fn, 'r') as handle: # loading all voices and assigning to instruments
            self.files_list = [x.strip().replace("_##WEIRDCHAR##_", '\n') for x in handle.readlines() if not x.startswith("#")]

        # self.files_list = ['/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Cantamus Catalog/Magnificat BWV 243 - 03 Quia respecit/voices/Soprano I.mp3', '/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Messies Participatiu/Gloria - 05. Domine Deus Rex celestis/voices/Alto.mp3', '/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Messies Participatiu/Gloria - 05. Domine Deus Rex celestis/voices/Bass.mp3', '/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Messies Participatiu/Gloria - 05. Domine Deus Rex celestis/voices/Tenor.mp3', '/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Messies Participatiu/Gloria - 09. Qui sedes ad dexteram Patris/voices/Bass.mp3', '/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Messies Participatiu/Gloria - 09. Qui sedes ad dexteram Patris/voices/Soprano.mp3', '/home/benedikt/thesis/datasets/CANTAMUS_SYNTHESIZED/Messies Participatiu/Gloria - 09. Qui sedes ad dexteram Patris/voices/Tenor.mp3']
        # #============================================
        # #validating potentially infinite files
        
        # files_remove_list = []
        # for fn in tqdm.tqdm(self.files_list):
        #     audio = load_wav_torch(fn, sr=sr, normed=True, reverted=False)[0]
        #     if not torch.isfinite(audio).all():
                
        #         files_remove_list.append(fn)
        #         print(f"already found not to be finite:", files_remove_list)

        # print("files to remove!")
        # pprint(files_remove_list)

        # for x in files_remove_list:
        #     self.files_list.remove(x)
        
        # print(f"#files removed: {len (files_remove_list)}")
        # print ("are they in files list?:", [x in self.files_list for x in files_remove_list])
        # #============================================

        files_df = [(Path(x), Path(x).parts[-4] + "__" + Path(x).parts[-3], Path(x).stem) 
                    for x 
                    in self.files_list]

        files_df = pd.DataFrame(files_df, columns = ["full_path", "short_path", "name"])
        files_df["voice_group"] = np.nan

        for index, row in files_df.iterrows():
            files_df.loc[index, "voice_group"] = get_voice_type(row["name"])
            
        print ("The following set of names will be dropped for unknown voice group")
        print (set (files_df[files_df["voice_group"].isnull()].name))
        files_df.drop(files_df[files_df["voice_group"].isnull()].index, inplace=True)

        #self.files_df = files_df

        self.grouped_by_voice_groups = dict (list (files_df.groupby(files_df["voice_group"])))
        self.grouped_by_piece = dict (list (files_df.groupby('short_path')))

        with open (rir_list_fn, 'r') as handle:
            self.rir_list = [x.strip() for x in handle.readlines()]
            
        with open (mir_list_fn, 'r') as handle:
            self.mir_list = [x.strip() for x in handle.readlines()]

        self.sr=sr
        self.eps=eps
        self.sample_length = int(sr*sub_sample_length)
        self.max_audio_length = int(max_audio_length*sr)

        self.nr_aug_variations=nr_aug_variations
        self.nr_samples_per_voice_v_augmentation_combination=nr_samples_per_voice_v_augmentation_combination
        self.mir_apply_cnt_list=mir_apply_cnt_list
        self.rir_apply_cnt_list=rir_apply_cnt_list
        
        self.aligned_prob=aligned_prob
        self.include_lead_in_accomp_prob=include_lead_in_accomp_prob

        self.sample_for_validation_or_test_creation=sample_for_validation_or_test_creation

        self.snr_list=self._parse_snr_range(snr_range)
        self.target_dB_FS=target_dB_FS
        self.target_dB_FS_floating_value=target_dB_FS_floating_value

        self.grvp_max_n_tries_foreground = grvp_max_n_tries_foreground
        self.grvp_max_n_tries_background = grvp_max_n_tries_background
        
        self.mix_buffer=[]


        self.dry_run=dry_run
        self.target_task=target_task

        self.silent_lead_prob = silent_lead_prob

        self.DEBUG_SINGULAR_BUFFER=DEBUG_SINGULAR_BUFFER

    def __len__(self):
        return 9999 #it can be infinite and in the case of sample_for_validation_or_test_creation there will be some dropped so depends...

    # def batch_augment (self, audio_batch=None, mir_list=None, rir_list=None, do_rir=False, do_mir=False, nr_aug_variations=0, sr=None):
    #     with torch.no_grad():
    #         if do_rir:
    #             curr_rir_list=rir_list
    #         elif do_mir:
    #             curr_rir_list=mir_list
    #         else:
    #             return None
            
    #         rir_choices = [curr_rir_list[i] for i in random.sample(range (len (curr_rir_list)), nr_aug_variations)]
    #         rir_audios = [load_wav_torch(fn, sr=sr, normed=True, reverted=True)[0] for fn in rir_choices]
    #         max_rir_len = max([rir_sample.shape[1] for rir_sample in rir_audios])

    #         rir_ = torch.stack([torch.nn.functional.pad(rir_sample, (max_rir_len-rir_sample.shape[1],0)) for rir_sample in rir_audios]).cuda()
    #         audio_batch = torch.nn.functional.pad(audio_batch, (max_rir_len-1, 0))
    #         audio_batch = torch.nn.functional.conv1d(audio_batch, rir_, groups=nr_aug_variations)
            
    #         del rir_
    #         return audio_batch

    # def create_augmented_batch (self,
    #                         voice_audios=None, 
    #                         mir_list=None, 
    #                         rir_list=None, 
    #                         rir_mir_queue=['rir', 'mir'], 
    #                         nr_aug_variations=0):
    #     with torch.no_grad():
    #         max_voice_len = max([voice_sample.shape[1] for voice_sample in voice_audios])
            
    #         singing = torch.stack([torch.nn.functional.pad(clean_y, (max_voice_len-clean_y.shape[1],0)) for clean_y in voice_audios.copy()]).cuda()
    #         dummy_convolve_ = torch.stack([torch.tensor([[1.0]]) for x in range(nr_aug_variations)]).cuda()
    #         augmented_batch = torch.nn.functional.conv1d(singing, dummy_convolve_)
            
            
    #         del dummy_convolve_
    #         #TODO don't do this! convolve rir's, mir's with themselves, then convolve with the whole thing! (grouping also not needed)
    #         for process_type in rir_mir_queue:
    #             if process_type == 'rir':
    #                 do_rir=True
    #                 do_mir=False
    #             elif process_type == 'mir':
    #                 do_rir=False
    #                 do_mir=True
    #             augmented_batch = self.batch_augment (augmented_batch,
    #                                         mir_list=mir_list,
    #                                         rir_list=rir_list,
    #                                         do_rir=do_rir,
    #                                         do_mir=do_mir,
    #                                         nr_aug_variations=nr_aug_variations)
                

    #         return singing.cpu(), augmented_batch.cpu()

    def zero_align_convolution_batch (self, tensor_in=None, debug=False, thresh=0.01): #0.01 means -40 dB on 0dB normalized signal
        individual_convolutions = [tensor_in[i,...,:] for i in range(tensor_in.shape[0])]
        
        del tensor_in #TODO NEEDED? works even?
        
        individual_convolutions_new = []
        
        for x in individual_convolutions:
            #scale to 0dBFS, see first sample above -40dBFS (or other, depends on definition)
            tmp = torch.abs (x)
            tmp = tmp / torch.max (tmp)
            
            first_voiced = torch.where(tmp / torch.max(tmp)>thresh)[1][0]
            
            individual_convolutions_new.append(x[..., first_voiced:])
            
            del tmp
            
        individual_convolutions = individual_convolutions_new
        max_len = max([sample.shape[-1] for sample in individual_convolutions])
        ret = torch.stack([torch.nn.functional.pad(sample, (0, max_len-sample.shape[-1])) for sample in individual_convolutions]).cuda()
        
        # if debug:
        #     batch_plot(ret)
        
        return ret, max_len



    def create_augmented_batch (self,
                                voice_audios=None, 
                                mir_list=None, 
                                rir_list=None, 
                                rir_mir_queue=['rir', 'mir'], 
                                nr_aug_variations=0,
                                sr = 16000):
        #start = time.time()
        with torch.no_grad():
            #start = time.time()
            
            max_voice_len = max([voice_sample.shape[1] for voice_sample in voice_audios])

            singing = torch.stack([torch.nn.functional.pad(clean_y, (0, max_voice_len-clean_y.shape[1])) for clean_y in voice_audios.copy()]).cuda()
            
            dummy_convolve_ = torch.stack([torch.tensor([[1.0]]) for x in range(nr_aug_variations)]).cuda()
            augmented_batch = torch.nn.functional.conv1d(singing, dummy_convolve_)
            
            if rir_mir_queue == []:
                rir_mir_queue = ['rir'] #hacky way to force a convolution. TODO give more transparent solution

        
            process_type_to_rir_source = {
                "mir": mir_list,
                "rir": rir_list
            }
            
            head_item, tail_items = rir_mir_queue[0], rir_mir_queue[1:]
            
            curr_rir_list = process_type_to_rir_source[head_item]
            rir_choices = [curr_rir_list[i] for i in random.sample(range (len (curr_rir_list)), nr_aug_variations)]                                                              #!!!
            rir_audios = [load_wav_torch(fn, sr=sr, normed=True, reverted=False)[0] for fn in rir_choices]
            max_rir_len = max([rir_sample.shape[1] for rir_sample in rir_audios])
            
            
            
            head_ = torch.stack([torch.nn.functional.pad(rir_sample, (0, max_rir_len-rir_sample.shape[1])) for rir_sample in rir_audios]).cuda()
            head_ = torch.transpose(head_, 0 , 1)
            rir_choices_combinations = [[x] for x in rir_choices]
            
            
            for idx, process_type in enumerate(tail_items):
                
                
                curr_rir_list = process_type_to_rir_source[process_type]
                match = False
                while not match:
                    rir_choices = [curr_rir_list[i] for i in random.sample(range (len (curr_rir_list)), nr_aug_variations)]
                    
                    for i in range(nr_aug_variations):
                        rir_choices_combinations[i].append (rir_choices[i])
                    
                    #next code block is to assure that the convolutions done with each other are unique.
                    #kind of tedious 
                    # - but I don't know if the mir_list, rir_list is actually of unique values, so cannot take care by simply "black"
                    # - I could just one-by-one and check if the values overlap, but that would lose me the random.sample method or be just as tedious
                    
                    nr_unique_entries_expected = 2 + idx
                    nr_unique_entries_per_combination = set([len(set (x)) for x in rir_choices_combinations])
                    if nr_unique_entries_per_combination == set([nr_unique_entries_expected]):
                        match = True
                    else:
                        for i in range(nr_aug_variations):
                            rir_choices_combinations[i].pop() #remove last iteration
            
                rir_audios = [load_wav_torch(fn, sr=sr, normed=True, reverted=True)[0] for fn in rir_choices]
                max_rir_len = max([rir_sample.shape[1] for rir_sample in rir_audios])
                rir_ = torch.stack([torch.nn.functional.pad(rir_sample, (max_rir_len-rir_sample.shape[1],0)) for rir_sample in rir_audios]).cuda()
                
                head_ = torch.nn.functional.pad (head_, (max_rir_len-1, 0))
                head_ = torch.nn.functional.conv1d (head_, rir_, groups=nr_aug_variations)
                
            rir_=head_
            del head_ #TODO needed?
            rir_ = torch.transpose(rir_, 0, 1)
            rir_, max_rir_len = self.zero_align_convolution_batch (tensor_in=rir_)
            
            rir_ = torch.flip (rir_, [2])
            
            augmented_batch = torch.nn.functional.pad(augmented_batch, (max_rir_len-1, 0))
            augmented_batch = torch.nn.functional.conv1d(augmented_batch, rir_, groups=nr_aug_variations)
            
            #print("convolution process took", time.time() - start)
            #print(f"for {augmented_batch.shape[0]} times {augmented_batch.shape[2]/16000} seconds of audio")
            
            del rir_
            
            #return singing.cpu(), augmented_batch.cpu()
            return singing.cpu(), augmented_batch.cpu() #should both be at CUDA
    
    def pop_random_mixed_constellation(self, grouped_by_voice_groups=None):
        
        fns_ret = []
        for voice_group in sorted(set(grouped_by_voice_groups.keys())):
            fn = random.choice (list (grouped_by_voice_groups[voice_group]['full_path']))
            fns_ret.append(fn)
        return sorted(set(grouped_by_voice_groups.keys())), fns_ret

    def pop_random_aligned_constellation(self, grouped_by_piece=None):
        
        piece_name = random.choice(list (grouped_by_piece.keys()))
        piece = grouped_by_piece[piece_name]

        voice_groups_occurrent = set (piece['voice_group'])

        if len(voice_groups_occurrent) == piece.shape[0]:
            ret = list (piece['voice_group']), list (piece['full_path']), piece_name
        else:
            # unique voices strat
            ret = []
            for voice_group in voice_groups_occurrent:
                if voice_group == 'accompaniment': 
                    #group_df = piece[piece['voice_group']==voice_group] #just use all the instruments there, doesn't matter
                    group_df = piece[piece['voice_group']==voice_group].sample(1) #actually just use one accompaniment to save resources
                    ret += list(zip(list(group_df.voice_group), list(group_df.full_path)))

                else: #dealing with vocal voice type
                    group_df = piece[piece['voice_group']==voice_group].sample(1)
                    ret += list(zip(list(group_df.voice_group), list(group_df.full_path)))
                    
            if len(ret) == 0:
                return None
            
            ret = list(zip(*ret))
            ret = ret[0], ret[1], piece_name
            
        return ret

    def is_voiced_simple(self,
                        sig=None, 
                        voiced_amplitude_threshold = 0.02,
                        voiced_ratio_threshold = 0.05,
                        rms_db_limit = -30,
                        ):
        if sig.count_nonzero() == 0:
            #warnings.warn("get_random_voiced_part: shouldn't happen. Dead soundfile...")
            return False
        
        # TODO check if it worked, but the normalization is done earlier!
        rms = torch.sqrt(torch.mean(sig ** 2))
        rms = 20*torch.log10(rms) # to dB, warning here is fine when we have rms=0 -> will give -inf rms

        voiced_ratio = (sig > voiced_amplitude_threshold).sum() / len(sig)

        if (rms > rms_db_limit and voiced_ratio > voiced_ratio_threshold):
            # print ("found after", i, "subsampling tries")
            return True
        else:
            return False

    def get_random_voiced_part(self,
                            sig=None, 
                            sr=-1, 
                            voiced_amplitude_threshold = 0.04, 
                            voiced_ratio_threshold = 0.05, 
                            rms_db_limit = -20, 
                            sample_length = -1, 
                            eps = 1e-6,
                            max_n_tries = 5 # TODO MAKE ALSO THE ABOVE AVAILABLE AS HYPERPARAMETER; ALSO THE ABOVE. DISTINCTION BETWEEN FOREGROUND/BACKGROUND REQUIREMENTS!!
                            ):
        """
        arguments:
        fn="",
        sr=-1, 
        voiced_amplitude_threshold = 0.02, 
        voiced_ratio_threshold = 0.05, 
        rms_db_limit = -30, 
        sample_length = -1, 
        eps = 1e-6,
        max_n_tries = 30
        
        returns:
        None if did not find a matching sample after max_n_tries
        OR
        subsampled, start_pos: signal as torch tensor and start position inside audio file
        """
        if sig.count_nonzero() == 0:
            #warnings.warn("get_random_voiced_part: shouldn't happen. Dead soundfile...")
            return None

        # TODO check if it worked, but the normalization is done earlier!
        #sig = sig / (torch.max(sig) + eps)
        
        for i in range (max_n_tries):
            
            
            subsampled, start_pos = subsample_audio_tensor(sig, sub_sample_length=sample_length, return_start_position=True)
            
            is_voiced = self.is_voiced_simple(sig=subsampled, 
                                        voiced_amplitude_threshold=voiced_amplitude_threshold,
                                        voiced_ratio_threshold=voiced_ratio_threshold,
                                        rms_db_limit=rms_db_limit
                                        )
            
            if is_voiced:
                return subsampled, start_pos
    #         rms = np.sqrt(np.mean(subsampled ** 2))
    #         rms = 20*np.log10(rms) # to dB
            
    #         voiced_ratio = (subsampled > voiced_amplitude_threshold).sum()/len(subsampled)
            
    #         if (rms > rms_db_limit and voiced_ratio > voiced_ratio_threshold):
    #             # print ("found after", i, "subsampling tries")
    #             return subsampled, start_pos

        #print ("no voiced part found for file")# , fn)
        return None

    def get_foreground_background_mixes (self,
                                        mode="aligned", 
                                        include_lead_in_accomp=True,
                                        voice_groups=None, 
                                        singing=None, 
                                        singing_augmented_batch=None,
                                        sample_length=None,
                                        nr_aug_variations=None,
                                        nr_samples_per_voice_v_augmentation_combination=None,
                                        sample_for_validation_or_test_creation=False,
                                        eps=1e-6,
                                        sr=None):
        
        
        leading_voices = set(voice_groups) - {'accompaniment'}
        leading_voices_idx = [voice_groups.index(x) for x in leading_voices]
        
        all_mixes=[]
        
        if sample_for_validation_or_test_creation: 
            #in this case we do not want to create combinations for the whole augmentation matrix (singing_augmented_batch)
            #we just want to use the identity
            assert nr_aug_variations >= len(leading_voices_idx) #so that we can get a unique augmentation for each voice
            assert nr_samples_per_voice_v_augmentation_combination == 1
            #assert mode=='aligned' #turn off if you want to allow test set creation with unaligned data
            
        for _ in range (nr_samples_per_voice_v_augmentation_combination):
            for i in leading_voices_idx:
                ignore_this_voice=False

                for j in range(nr_aug_variations):

                    if sample_for_validation_or_test_creation: 
                        if i != j:
                            continue
                    if ignore_this_voice:
                        continue
                    # print (i,j)

                    if include_lead_in_accomp:
                        accompaniment_idxs=list(range(len(voice_groups)))
                    else:
                        accompaniment_idxs = [x for x in range(len(voice_groups)) if x != i]


                    leading_voice_group = voice_groups[i]


                    sig_lead = singing[i, 0, :] #clean singing...
                    _ = self.get_random_voiced_part(sig_lead, sr=sr, sample_length=sample_length, max_n_tries = self.grvp_max_n_tries_foreground) #find voiced excerpt of length sample_length
                    if _ == None:
                        ignore_this_voice=True
                        continue #disregard this voice if there seems to be no voiced part available

                    sig_lead, start_pos = _
                    #sig_lead is snippet of 0dB-normalized whole excerpt

                    if mode == 'aligned': #all parts are synthesized from same piece -> aligned, harmonically coherent voices
                        sig_accompaniments = []
                        sig_accompaniments_orig = [] #original background music

                        for x in accompaniment_idxs:
                            #sig, _ = norm_amplitude (singing_augmented_batch[x, j, :]) # TODO double check, but normalization should have happened in beginning
                            sig = singing_augmented_batch[x, j, :]
                            sig_accompaniments.append (sig[start_pos:start_pos+sample_length])
                            #sig, _ = norm_amplitude (singing[x, 0, :])
                            sig = singing[x, 0, :]
                            sig_accompaniments_orig.append (sig[start_pos:start_pos+sample_length])

                        #print (sig_accompaniments)
                        #code to drop too short excerpts that might be produced (shouldn't happen with cantamus synthesis)
                        # idx_v_sig = list(zip(accompaniment_idxs, sig_accompaniments, sig_accompaniments_orig))
                        # idx_v_sig = [(x, y, z) for x, y, z in idx_v_sig if ((y.shape == sig_lead.shape) and (z.shape == sig_lead.shape))]
                        # accompaniment_idxs, sig_accompaniments, sig_accompaniments_orig =  zip(*idx_v_sig) #inform about dropped excerpts
                        #sig_accompaniments elements are snippet of 0dB-normalized whole excerpt

                    else: #all parts are thrown together randomly
                        #after this statement, sig_accompaniments holds output of get_random_voice_part -> either tuple or None
                        sig_accompaniments = [
                            self.get_random_voiced_part(
                                singing_augmented_batch[x, j, :], 
                                sr=sr, 
                                sample_length=sample_length, 
                                max_n_tries = self.grvp_max_n_tries_background) 
                            for x 
                            in accompaniment_idxs
                        ]

                        idx_v_sig = list(zip(accompaniment_idxs, sig_accompaniments))
                        idx_v_sig = [(x, y) for x, y in idx_v_sig if y != None] #filter unvoiced excerpts
                        
                        if idx_v_sig==[]:
                            print ("didn't find a single voiced accompaniment for", leading_voice_group)
                            continue

                        accompaniment_idxs, sig_accompaniments =  zip(*idx_v_sig) #inform about dropped excerpts

                        sig_accompaniments_orig = [] #original background music
                        for x, accomp in idx_v_sig:
                            #sig, _ = norm_amplitude (singing[x, 0, :]) #TODO correctly removed?
                            sig = singing[x, 0, :]
                            start_pos = accomp[1] #align with sig_accompaniments
                            sig_accompaniments_orig.append (sig[start_pos:start_pos+sample_length])

                        sig_accompaniments = [x for x,_ in sig_accompaniments] #dropping the start position y, just use signal x

                        if include_lead_in_accomp and (i in accompaniment_idxs): #one should imply the other, but building safety net for future changes
                            # print("rare occasion?!")
                            # print(sig_accompaniments)
                            overwrite_idx = accompaniment_idxs.index(i)
                            sig = singing_augmented_batch[i, j, :]
                            #sig, _ = norm_amplitude (sig) #TODO correctly removed? #important: 0dB-normalize each signal
                            # print(sig)
                            # print(sig[start_pos:start_pos+sample_length])
                            sig_accompaniments[overwrite_idx] = sig[start_pos:start_pos+sample_length]
                            # print("AFTER?!")
                            # print(sig_accompaniments)

                    accompaniment_voice_groups = [voice_groups[x] for x in accompaniment_idxs]

                    all_mixes.append((leading_voice_group, sig_lead, accompaniment_voice_groups, sig_accompaniments, sig_accompaniments_orig))
        
        return all_mixes

    def snr_mix (self, mix_input=None, sr=16000, sample_length=None, debug=False, snr=None, eps=1e-6, target_dB_FS=None, target_dB_FS_floating_value=None, current_piece_info="", mode="", silent_lead_prob=0.005): 
        leading_voice_group, sig_lead, accompaniment_voice_groups, sig_accompaniments, sig_accompaniments_orig = mix_input
        
        piece_id = current_piece_info + "__" + leading_voice_group

        if accompaniment_voice_groups != []:
            nr_accomp = random.choice(range(0, len(accompaniment_voice_groups) + 1))
        else:
            nr_accomp=0
        #print(f"nr of accompaniment voices is: {nr_accomp} of {len(accompaniment_voice_groups)}")
        
        acc_voice_v_acc_sig = random.sample(list(zip(accompaniment_voice_groups, sig_accompaniments, sig_accompaniments_orig)), nr_accomp)
        
        sig_lead, _, _ = tailor_dB_FS(sig_lead, target_dB_FS) #TODO is this needed at all?
        sig_lead_rms = (sig_lead ** 2).mean() ** 0.5
        
        noisy_target_dB_FS = np.random.randint(
                target_dB_FS - target_dB_FS_floating_value,
                target_dB_FS + target_dB_FS_floating_value
            )
        
        if acc_voice_v_acc_sig == []: #no accompaniment
            #print(f"{mode}: no accompaniment for {piece_id}")
            accomp_mix = sig_lead*0.0
            accomp_mix_orig = sig_lead*0.0
            sig_lead, _, _ = tailor_dB_FS(sig_lead, noisy_target_dB_FS)
            mixture = sig_lead
            # print("only zeros in mix... pass_through")
            
        else:    
            accompaniment_voice_groups, sig_accompaniments, sig_accompaniments_orig = zip(*acc_voice_v_acc_sig)
            #print("accomp dimensions", sig_accompaniments.shape)
            
            try:
                sig_accompaniments = torch.stack(sig_accompaniments)
            except TypeError:
                print(sig_accompaniments)
                print(accompaniment_voice_groups)
                print(sig_accompaniments_orig)

            sig_accompaniments_orig = torch.stack(sig_accompaniments_orig)

            accomp_mix = torch.sum(sig_accompaniments, axis = 0)
            #print("accomp mix dimensions", accomp_mix.shape)

            if self.is_voiced_simple(accomp_mix, voiced_amplitude_threshold = 0.01, voiced_ratio_threshold = 0.02, rms_db_limit = -35): #TODO apply special settings for is_voiced_simple from config
                #print(f"{mode}:accompaniment EXISTS for {piece_id}")
                accomp_mix,_ = norm_amplitude(accomp_mix)
                accomp_mix, _, _ = tailor_dB_FS(accomp_mix, target_dB_FS) #TODO check do we need this line of code? I suppose not
                accomp_mix_rms = (accomp_mix ** 2).mean() ** 0.5

                

                snr_scalar = sig_lead_rms / (10 ** (snr / 20)) / (accomp_mix_rms + eps)

                accomp_mix *= snr_scalar

                mixture = accomp_mix + sig_lead

                mixture, _, noisy_scalar = tailor_dB_FS(mixture, noisy_target_dB_FS)
                accomp_mix *= noisy_scalar
                sig_lead *= noisy_scalar

                if is_clipped(mixture):
                    mixture, noisy_y_scalar = norm_amplitude (mixture)
                    sig_lead = sig_lead / noisy_y_scalar
                    accomp_mix = accomp_mix / noisy_y_scalar


                accomp_mix_orig = torch.sum (sig_accompaniments_orig, axis=0) #amount of bleed must be guessed, so this is just normed
                #print("accomp mix orig dimensions", accomp_mix_orig.shape)
                if is_clipped(accomp_mix_orig):
                    accomp_mix_orig, _ = norm_amplitude(accomp_mix_orig)
            else:
                #print(f"{mode}: no accompaniment for {piece_id}")
                
                sig_lead, _, _ = tailor_dB_FS(sig_lead, noisy_target_dB_FS)
                mixture = sig_lead + accomp_mix
                accomp_mix_orig = torch.sum (sig_accompaniments_orig, axis=0) #DO NOT NORMALIZE THIS AS IT IS VERY LIKELY SILENT SIGNAL
                #print("ententent", torch.max(torch.abs(accomp_mix_orig)))
            # else: likely to have only silent parts -> we do not want to norm those, it would produce noise! -> do nothing


        if debug:
            print (leading_voice_group, "accompanied by", accompaniment_voice_groups)
            print ("snr is ", snr)
            # display(ipd.Audio(sig_lead, rate=sr, normalize=False))
            # display(ipd.Audio(mixture, rate=sr, normalize=False))
            # display(ipd.Audio(accomp_mix, rate=sr, normalize=False))
            # display(ipd.Audio(accomp_mix_orig, rate=sr, normalize=False))

        silent_lead = random.random() < silent_lead_prob
        if silent_lead:
            sig_lead = torch.zeros_like(sig_lead)
            mixture = accomp_mix

        if not torch.isclose(mixture, sig_lead + accomp_mix).all():
            abs_diff=torch.abs(mixture - (sig_lead+accomp_mix))
            print("mixture and sig_lead+accomp_mix DEVIATING!! by: (max deviation, sum absolute deviation)", torch.max(abs_diff), torch.sum(abs_diff))

        
        
        return sig_lead, mixture, accomp_mix, accomp_mix_orig, piece_id, snr
    
    
    def fill_buffer(self):
        
        
        if random.random() < self.aligned_prob:
            mode='aligned'
        else:
            mode='UNaligned'
        
        if random.random() < self.include_lead_in_accomp_prob:
            include_lead_in_accomp=True
        else:
            include_lead_in_accomp=False

        rir_mir_queue=['mir']*random.choice(self.mir_apply_cnt_list) + ['rir']*random.choice(self.rir_apply_cnt_list)

        print ("rir_mir_queue is:", rir_mir_queue)
        print ('mode is:', mode)

        if mode == 'aligned':
            assert len (self.grouped_by_piece) > 0 #see pop operation some in case of sample_for_validation_or_test_creation
            voice_groups, fns, piece = self.pop_random_aligned_constellation(self.grouped_by_piece)
            if self.sample_for_validation_or_test_creation: #make sure we visit every piece only once
                self.grouped_by_piece.pop (piece)
        else:
            voice_groups, fns = self.pop_random_mixed_constellation(self.grouped_by_voice_groups)

        voice_audios = [load_wav_torch(fn, sr=self.sr, normed=True, reverted=False)[0] for fn in fns]
        # norm instantly!!! important when we later apply max_audio_length as there might be solely unvoiced parts for some files before cutoff -> if we norm later, we receive noise...
        print ("instant norming")
        voice_audios = [norm_amplitude(x, soft_fail=True)[0] for x in voice_audios]

        for fn, audio in zip(fns, voice_audios):
            if type(audio)==type(None):
                print(f"DEAD FILE: {fn}")
            elif audio.shape[-1] < self.sample_length:
                print(f"VERY SHORT FILE DROPPED: {fn}")

        fn_v_audio_v_vg = [(fn, audio, vg) for fn, audio, vg in zip(fns, voice_audios, voice_groups) if type(audio)!=type(None)]
        fns, voice_audios, voice_groups = zip(*[(fn, audio, vg) for fn, audio, vg in fn_v_audio_v_vg if audio.shape[-1] >= self.sample_length])
        fns = list(fns)
        voice_audios = list(voice_audios)
        voice_groups = list(voice_groups)

        

        audio_lengths = np.array([x.shape[1] for x in voice_audios])
        max_occurred_length = int(np.max(audio_lengths))
        min_occurred_length = int(np.min(audio_lengths))

        # print ("longest audio is:", int(max_occurred_length))
        # print ("shortest audio is:", int(min_occurred_length))

        excerpt_trim_limit = min(min_occurred_length, self.max_audio_length)
        # print ("excerpt trim limit is", excerpt_trim_limit)
        start_pos = np.random.randint(min_occurred_length - excerpt_trim_limit + 1) # +1 because right border is excluded

        voice_audios = [ x[:, start_pos : start_pos + excerpt_trim_limit] for x in voice_audios ]
        audio_lengths = np.array([x.shape[1] for x in voice_audios])
        print (audio_lengths)

        for audio, fn in zip(voice_audios, fns):
            assert torch.isfinite(audio).all(), f"{fn} is infinite!!"

        print ("sounds are of voice types:", voice_groups)

        if self.dry_run:
            self.mix_buffer=[1]
            return 

        with torch.no_grad():
            start = time.time()
            singing, singing_augmented_batch = self.create_augmented_batch(voice_audios=voice_audios, 
                                                mir_list=self.mir_list, 
                                                rir_list=self.rir_list, 
                                                rir_mir_queue=rir_mir_queue, 
                                                nr_aug_variations=self.nr_aug_variations,
                                                sr=self.sr)
            end = time.time()
            print ("convolution augmentation took:", end-start, "seconds")
            assert torch.isfinite(singing).all(), "Sining was infinite!!!!"
            assert torch.isfinite(singing_augmented_batch).all(), "augmented singing was infinite!!!"

            start = time.time()
            current_buffered_mixes=self.get_foreground_background_mixes (mode=mode, 
                                                            include_lead_in_accomp=include_lead_in_accomp,
                                                            voice_groups=voice_groups, 
                                                            singing=singing, 
                                                            singing_augmented_batch=singing_augmented_batch,
                                                            sample_length=self.sample_length,
                                                            nr_aug_variations=self.nr_aug_variations,
                                                            nr_samples_per_voice_v_augmentation_combination=self.nr_samples_per_voice_v_augmentation_combination,
                                                            sample_for_validation_or_test_creation=self.sample_for_validation_or_test_creation,
                                                            sr=self.sr)
            print (f"{singing.shape[0]*singing_augmented_batch.shape[1]*self.nr_samples_per_voice_v_augmentation_combination} iterations produced {len(current_buffered_mixes)} foreground-background combinations")
            end = time.time()
            print ("retrieving mix combinations and subsamples took:", end-start, "seconds in", mode, "mode")


            start = time.time()
            for mix_input in current_buffered_mixes:
                if mode == 'aligned':
                    current_piece = piece
                else:
                    current_piece = 'unknown'

                self.mix_buffer.append( self.snr_mix(   mix_input=mix_input, 
                                                        sr=self.sr, 
                                                        snr = random.choice (self.snr_list),
                                                        sample_length=self.sample_length, 
                                                        target_dB_FS=self.target_dB_FS,
                                                        target_dB_FS_floating_value=self.target_dB_FS_floating_value,
                                                        current_piece_info=current_piece,
                                                        mode=mode,
                                                        silent_lead_prob=self.silent_lead_prob,
                                                    ))
                
            print(f"{len(current_buffered_mixes)} foreground background combinations produced {len(self.mix_buffer)} valid mixes")
            end = time.time()
            print ("doing snr-mixes took:", end-start, "seconds")

            del singing
            del singing_augmented_batch
        
        random.shuffle (self.mix_buffer)
        return
    
    def pop_from_buffer(self):
        while self.mix_buffer == []:
            print("buffer is empty!")
            self.fill_buffer()

        if self.DEBUG_SINGULAR_BUFFER:
            return random.choice(self.mix_buffer)

        return self.mix_buffer.pop ()

    def __getitem__(self, item):
        sig_lead, mixture, accomp_mix, accomp_mix_orig, piece_id, snr = self.pop_from_buffer ()
        if self.target_task == "standard":
            return sig_lead, mixture, accomp_mix, accomp_mix_orig, piece_id, snr
        if self.target_task == "baseline_fsn":
            return mixture, sig_lead
        if self.target_task == "fsn_aec":
            return sig_lead, mixture, accomp_mix_orig