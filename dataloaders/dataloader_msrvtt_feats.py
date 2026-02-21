from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle5 as pickle
import pandas as pd
from collections import defaultdict
import json
import random
from tqdm import tqdm
from scipy import sparse
import glob
import h5py
from dataset.tokenizer_nv import WordIDMapper
class MSRVTT_Feats_DataLoader(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the model training and evaluation.
    Params:
        json_path: Path to the MSRVTT_data.json file.
        features_path: Path to the extracted feature file.
        tokenizer: Tokenizer used for tokenizing the caption.
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        split_type: Either "train", "val", or "test". Default: ""
    """
    def __init__(
            self,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            video_feat_type=""
    ):
        self.data = json.load(open(json_path, 'r'))

        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_swin = h5py.File("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/extracted_feats/msrvtt/motion_swinbert_kinetics_cliplen64_dense.hdf5", 'r')
        self.tokenizer_v = WordIDMapper("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/dataset/MSVD/vocab_v.txt").word_to_id
        self.tokenizer_n = WordIDMapper("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/dataset/MSVD/vocab_n.txt").word_to_id

        v_n_dic_path = os.path.join(json_path[:-16], "V_N_MSRVTT.json")
        with open(v_n_dic_path, 'r') as f:
            v_n_dic = json.load(f)
        self.video_feat_type = video_feat_type
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.feature_size = self.feature_dict[next(iter(self.feature_dict))].shape[-1]
        #split_type =
        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict_ = defaultdict(list)
        self.sentences_dict = defaultdict(list)
        self.video_sentences_dict = defaultdict(list)

        if split_type == "train":  # expand all sentence to train
            # for itm in self.data['sentences']:
            #     if itm['video_id'] in choiced_video_ids:
            #         self.sentences_dict_[len(self.sentences_dict_)] = (itm['video_id'], itm['caption']) #{0:(vid,caption)}
            #         self.video_sentences_dict[itm['video_id']].append(itm['caption'])  # {vid:[caption1,caption2]}
            for itm_v_n in v_n_dic:
                if itm_v_n['video_id'] in choiced_video_ids:
                    self.sentences_dict_[len(self.sentences_dict_)] = (itm_v_n['video_id'], itm_v_n['caption'],itm_v_n['verbs'],itm_v_n['nouns'])  # {0:(vid,caption)}
                    self.video_sentences_dict[itm_v_n['video_id']].append([itm_v_n['caption'],itm_v_n['verbs'],itm_v_n['nouns']])  # {vid:[caption1,caption2]}
            #video_id, caption,caption_pos,caption_neg ,cap_verbs, cap_nouns, cap_pos_verbs, cap_pos_nouns, cap_neg_verbs, cap_neg_nouns
            keys_list = list(self.video_sentences_dict.keys())
            for key in self.sentences_dict_:

                caption_pos_all = random.choice(self.video_sentences_dict[self.sentences_dict_[key][0]])
                caption_pos, cap_pos_verbs, cap_pos_nouns =  caption_pos_all[0],caption_pos_all[1],caption_pos_all[2]
                while caption_pos == self.sentences_dict_[key][1]:
                    caption_pos_all = random.choice(self.video_sentences_dict[self.sentences_dict_[key][0]])##
                    caption_pos, cap_pos_verbs, cap_pos_nouns = caption_pos_all[0], caption_pos_all[1], caption_pos_all[2]

                vid_neg = random.choice(keys_list)
                while vid_neg == self.sentences_dict_[key][0]:
                    vid_neg = random.choice(keys_list)
                caption_neg_all = random.choice(self.video_sentences_dict[vid_neg])
                caption_neg,cap_neg_verbs, cap_neg_nouns = caption_neg_all[0],caption_neg_all[1],caption_neg_all[2]
                self.sentences_dict[key] = (self.sentences_dict_[key][0],self.sentences_dict_[key][1], caption_pos, caption_neg,self.sentences_dict_[key][2], self.sentences_dict_[key][3], cap_pos_verbs, cap_pos_nouns, cap_neg_verbs, cap_neg_nouns)
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0],'','','','','','','','')
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)


            
    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None, caption_pos =None, caption_neg=None,cap_verbs=None, cap_nouns=None, cap_pos_verbs=None, cap_pos_nouns=None, cap_neg_verbs=None, cap_neg_nouns=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_input_caption_ids_pos = np.zeros((k, self.max_words), dtype=np.long)
        pairs_input_caption_ids_neg = np.zeros((k, self.max_words), dtype=np.long)

        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids_pos = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids_neg = np.zeros((k, self.max_words), dtype=np.long)

        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask_pos = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask_neg = np.zeros((k, self.max_words), dtype=np.long)

        cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_ = \
            (
            np.zeros((k, 3), dtype=np.long),#self.max_words
            np.zeros((k, 5), dtype=np.long),#self.max_words
            np.zeros((k, 3), dtype=np.long),#self.max_words
            np.zeros((k, 5), dtype=np.long),#self.max_words
            np.zeros((k, 3), dtype=np.long),#self.max_words
            np.zeros((k, 5), dtype=np.long)#self.max_words
            )



        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]


            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
            assert len(input_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
                caption_words_pos = self.tokenizer.tokenize(caption_pos)
                caption_words_neg = self.tokenizer.tokenize(caption_neg)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            if len(caption_words_pos) > total_length_with_CLS:
                caption_words_pos = caption_words_pos[:total_length_with_CLS]
            if len(caption_words_neg) > total_length_with_CLS:
                caption_words_neg = caption_words_neg[:total_length_with_CLS]

            caption_words_ = caption_words
            if len(caption_words) > total_length_with_CLS-1:
                caption_words_ = caption_words[:total_length_with_CLS-1]
            input_caption_words = ["[CLS]"] + caption_words
            input_caption_words_pos = ["[CLS]"] + caption_words_pos
            input_caption_words_neg = ["[CLS]"] + caption_words_neg
            output_caption_words = caption_words + ["[SEP]"]
            output_caption_words_pos = caption_words_pos + ["[SEP]"]
            output_caption_words_neg = caption_words_neg + ["[SEP]"]

            single_caption_words = ["[CLS]"] + caption_words_ + ["[SEP]"]
            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            input_caption_ids_pos = self.tokenizer.convert_tokens_to_ids(input_caption_words_pos)
            output_caption_ids_pos = self.tokenizer.convert_tokens_to_ids(output_caption_words_pos)
            input_caption_ids_neg = self.tokenizer.convert_tokens_to_ids(input_caption_words_neg)

            output_caption_ids_neg = self.tokenizer.convert_tokens_to_ids(output_caption_words_neg)

            single_caption_ids = self.tokenizer.convert_tokens_to_ids(single_caption_words)

            decoder_mask = [1] * len(input_caption_ids)
            decoder_mask_pos = [1] * len(input_caption_ids_pos)
            decoder_mask_neg = [1] * len(input_caption_ids_neg)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            while len(output_caption_ids_pos) < self.max_words:
                input_caption_ids_pos.append(0)
                output_caption_ids_pos.append(0)
                decoder_mask_pos.append(0)
            while len(output_caption_ids_neg) < self.max_words:
                input_caption_ids_neg.append(0)
                output_caption_ids_neg.append(0)
                decoder_mask_neg.append(0)

            while len(single_caption_ids) < self.max_words:
                single_caption_ids.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(input_caption_ids_pos) == self.max_words
            assert len(input_caption_ids_neg) == self.max_words

            assert len(output_caption_ids) == self.max_words
            assert len(output_caption_ids_pos) == self.max_words
            assert len(output_caption_ids_neg) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_input_caption_ids_pos[i] = np.array(input_caption_ids_pos)
            pairs_input_caption_ids_neg[i] = np.array(input_caption_ids_neg)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)

            pairs_decoder_mask[i] = np.array(decoder_mask)
            pairs_decoder_mask_pos[i] = np.array(decoder_mask_pos)
            pairs_decoder_mask_neg[i] = np.array(decoder_mask_neg)


            cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_ = \
                (
                    np.zeros((k, 3), dtype=np.long),  # self.max_words
                    np.zeros((k, 5), dtype=np.long),  # self.max_words
                    np.zeros((k, 3), dtype=np.long),  # self.max_words
                    np.zeros((k, 5), dtype=np.long),  # self.max_words
                    np.zeros((k, 3), dtype=np.long),  # self.max_words
                    np.zeros((k, 5), dtype=np.long)  # self.max_words
                )

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids,pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
               cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)

        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        return video, video_mask, np.array([]), np.array([])

    def _get_video_swin(self, choice_video_ids):
        #print(choice_video_ids)

        video_mask = np.zeros((len(choice_video_ids), 1568), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), 1568, 1024), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            #print(video_id)
            video_slice = self.feature_swin[video_id][:]

            if 1568 < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        return video, video_mask, np.array([]), np.array([])

    def __getitem__(self, idx):
        video_id, caption,caption_pos,caption_neg ,cap_verbs, cap_nouns, cap_pos_verbs, cap_pos_nouns, cap_neg_verbs, cap_neg_nouns = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids,pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
        cap_verbs_ids_, cap_nouns_ids_,cap_pos_verbs_ids_, cap_pos_nouns_ids_,cap_neg_verbs_ids_, cap_neg_nouns_ids_ = self._get_text(video_id, caption, caption_pos , caption_neg,cap_verbs, cap_nouns, cap_pos_verbs, cap_pos_nouns, cap_neg_verbs, cap_neg_nouns)

        if self.video_feat_type=='Swin':
            video, video_mask, masked_video, video_labels_index = self._get_video_swin(choice_video_ids)
        elif self.video_feat_type=='Swin+Clip':
            video_swin, video_mask_swin, masked_video_swin, video_labels_index_swin = self._get_video_swin(choice_video_ids)
            video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        else:
            video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        

        pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

        # pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,
        # input_mask=pairs_mask, segment_ids=pairs_segment
        # the above data are unnecessary to train clip4caption
        if self.video_feat_type == 'Swin+Clip':
            return pairs_text, \
                pairs_mask, pairs_segment, \
                video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                video_swin, video_mask_swin, masked_video_swin, video_labels_index_swin, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, pairs_decoder_mask_pos, pairs_decoder_mask_neg, pairs_input_caption_ids_pos, pairs_output_caption_ids_pos, pairs_input_caption_ids_neg, pairs_output_caption_ids_neg, \
                cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_

        else:
            return pairs_text, \
                pairs_mask, pairs_segment, \
                video, video_mask, \
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, pairs_decoder_mask_pos, pairs_decoder_mask_neg, pairs_input_caption_ids_pos, pairs_output_caption_ids_pos, pairs_input_caption_ids_neg, pairs_output_caption_ids_neg, \
                cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_
