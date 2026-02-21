from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pdb

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from tqdm import tqdm
import glob
from scipy import sparse
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import nltk
import pickle5 as pickle
import h5py
import os
from dataset.tokenizer_nv import WordIDMapper
class MSVD_Feats_DataLoader(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the model training and evaluation.
    Params:
        data_path: Path to the MSVD folder.
        features_path: Path to the extracted feature file.
        tokenizer: Tokenizer used for tokenizing the caption.
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        split_type: Either "train", "val", or "test". Default: ""
    """

    def __init__(
            self,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            video_feat_type="",
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_swin = h5py.File("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/extracted_feats/msvd/motion_swinbert_kinetics_cliplen64_dense_my.hdf5", 'r')
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.video_feat_type = video_feat_type

        assert split_type in ["train", "val", "test"]
        self.tokenizer_v = WordIDMapper("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/dataset/MSVD/vocab_v.txt").word_to_id
        self.tokenizer_n = WordIDMapper("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/dataset/MSVD/vocab_n.txt").word_to_id

        split_dict = {}
        # video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict["train"] = os.path.join(self.data_path, "train_list_mapping.txt")
        split_dict["val"] = os.path.join(self.data_path, "val_list_mapping.txt")
        split_dict["test"] = os.path.join(self.data_path, "test_list_mapping.txt")
        caption_file = os.path.join(self.data_path, "raw-captions_mapped.pkl")
        self.feature_size = self.feature_dict['vid1'].shape[-1]

        # v_n_dic
        v_n_dic_path = os.path.join(self.data_path, "V_N_MSVD.json")
        v_n_dic = json.load(open(v_n_dic_path, 'r'))

        #sour
        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        # with open("/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/dataset/MSVD/V_N_MSVD.json", "r") as f:
        #     captions= json.load(f)

        with open(split_dict[split_type], 'r') as fp:
            choiced_video_ids = [itm.strip() for itm in fp.readlines()]
        # choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            index1 = 0
            length1 = len(captions)
            keys = list(captions.keys())
            for video_id in captions:
                if video_id in choiced_video_ids:
                    index2 = 0
                    for cap in captions[video_id]:
                        length2 = len(captions[video_id])
                        cap_pos = captions[video_id][random.choice([i for i in range(0, length2) if i != index2])]
                        neg_vid = random.choice([i for i in range(0, length1) if i != index1])
                        cap_neg = captions[keys[neg_vid]][random.randint(0, len(captions[keys[neg_vid]])-1)]

                        # cap_txt = cap[0]
                        # cap_text_pos = cap_pos[0]
                        # cap_text_neg = cap_neg[0]

                        cap_txt = " ".join(cap)
                        cap_text_pos = " ".join(cap_pos)
                        cap_text_neg = " ".join(cap_neg)


                        #cap_verbs_ids, cap_nouns_ids = self.get_N_V2ids(cap_txt)
                        cap_verbs, cap_nouns  = v_n_dic[video_id][index2][1],v_n_dic[video_id][index2][2]
                        #cap_pos_verbs_ids, cap_pos_nouns_ids = self.get_N_V2ids(cap_text_pos)
                        cap_pos_verbs, cap_pos_nouns = v_n_dic[video_id][random.choice([i for i in range(0, length2) if i != index2])][1],v_n_dic[video_id][random.choice([i for i in range(0, length2) if i != index2])][2]
                        #cap_neg_verbs, cap_neg_nouns = self.get_N_V2ids(cap_text_neg)
                        cap_neg_verbs, cap_neg_nouns = v_n_dic[keys[neg_vid]][random.randint(0, len(captions[keys[neg_vid]])-1)][1],v_n_dic[keys[neg_vid]][random.randint(0, len(captions[keys[neg_vid]])-1)][2]
                        self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt, cap_text_pos,cap_text_neg,cap_verbs[0], cap_nouns[0], cap_pos_verbs[0], cap_pos_nouns[0], cap_neg_verbs[0], cap_neg_nouns[0])
                        self.video_sentences_dict[video_id].append(cap_txt)
                        index2 += 1
                index1 += 1
        elif split_type == "val" or split_type == "test":
            for itm in captions:
                if itm in choiced_video_ids:
                    for cap in captions[itm]:

                        #cap_txt = cap[0]

                        cap_txt = " ".join(cap)
                        self.video_sentences_dict[itm].append(cap_txt)
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0],'','','','','','','','')
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)
    #
    # def get_N_V2ids(self,sentence_str):
    #      pattern = [[{'POS': 'VERB', 'OP': '?'},
    #                  {'POS': {'NOT_IN': ['DET', 'CCONJ', 'ADP', 'PRON', 'ADV']}, 'OP': '*', 'IS_STOP': True},  # 排除非停用词
    #                  {'POS': 'AUX', 'OP': '*'},
    #                  {'POS': 'VERB', 'OP': '+'}]]
    #      # 实例化一个 Matcher 实例
    #      nlp = spacy.load('en_core_web_sm')
    #      matcher = Matcher(nlp.vocab)
    #      # 将模式添加到 Matcher
    #      matcher.add("Verb phrase", pattern)
    #      doc = nlp(sentence_str)
    #      # 调用 Matcher 找到匹配项
    #      matches = matcher(doc)
    #      spans = [doc[start:end] for _, start, end in matches]
    #      filter_verbs = filter_spans(spans)
    #      filter_out_strings = [str(token) for token in filter_verbs]
    #      result_verbs = ' '.join(filter_out_strings)
    #      #print('动词', verbs)
    #      ##名词
    #      is_noun = lambda pos: pos[:2] == 'NN'
    #      tokenized = nltk.word_tokenize(sentence_str)
    #      nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    #      reuslt_nouns = ' '.join(nouns)
    #      #print('名词', nouns)
    #      if len(result_verbs) > 6:
    #          result_verbs = result_verbs[:6]
    #      if len(reuslt_nouns) > 12:
    #          reuslt_nouns = reuslt_nouns[:12]
    #
    #      result_verbs_token = self.tokenizer.tokenize(result_verbs)
    #      result_verbs_ids = self.tokenizer.convert_tokens_to_ids(result_verbs_token)
    #      reuslt_nouns_token = self.tokenizer.tokenize(reuslt_nouns)
    #      reuslt_nouns_ids = self.tokenizer.convert_tokens_to_ids(reuslt_nouns_token)
    #
    #      while len(result_verbs_ids) < 6:
    #          result_verbs_ids.append(0)
    #      while len(reuslt_nouns_ids) < 12:
    #          reuslt_nouns_ids.append(0)
    #      return result_verbs_ids,reuslt_nouns_ids

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

        cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_ = \
            (
            np.zeros((k, 3), dtype=np.long),#self.max_words
            np.zeros((k, 5), dtype=np.long),#self.max_words
            np.zeros((k, 3), dtype=np.long),#self.max_words
            np.zeros((k, 5), dtype=np.long),#self.max_words
            np.zeros((k, 3), dtype=np.long),#self.max_words
            np.zeros((k, 5), dtype=np.long)#self.max_words
            )


        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask_pos = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask_neg = np.zeros((k, self.max_words), dtype=np.long)

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
            assert len(output_caption_ids) == self.max_words
            assert len(output_caption_ids_pos) == self.max_words
            assert len(input_caption_ids_neg) == self.max_words
            assert len(output_caption_ids_neg) == self.max_words

            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_input_caption_ids_pos[i] = np.array(input_caption_ids_pos)
            pairs_output_caption_ids_pos[i] = np.array(output_caption_ids_pos)
            pairs_input_caption_ids_neg[i] = np.array(input_caption_ids_neg)
            pairs_output_caption_ids_neg[i] = np.array(output_caption_ids_neg)


            pairs_decoder_mask[i] = np.array(decoder_mask)
            pairs_decoder_mask_pos[i] = np.array(decoder_mask_pos)
            pairs_decoder_mask_neg[i] = np.array(decoder_mask_neg)

            if cap_verbs != '':
                if len(cap_verbs) > 3:
                    cap_verbs = cap_verbs[:3]
                if len(cap_nouns) > 5:
                    cap_nouns = cap_nouns[:5]
                if len(cap_pos_verbs) > 3:
                    cap_pos_verbs = cap_pos_verbs[:3]
                if len(cap_pos_nouns) > 5:
                    cap_pos_nouns = cap_pos_nouns[:5]
                if len(cap_neg_verbs) > 3:
                    cap_neg_verbs = cap_neg_verbs[:3]
                if len(cap_neg_nouns) > 5:
                    cap_neg_nouns = cap_neg_nouns[:5]
                cap_verbs_ids = self.tokenizer_v(cap_verbs)
                cap_nouns_ids = self.tokenizer_n(cap_nouns)

                #cap_verbs_token = self.tokenizer.tokenize(cap_verbs)
                #cap_verbs_ids = self.tokenizer.convert_tokens_to_ids(cap_verbs_token)
                #cap_nouns_token = self.tokenizer.tokenize(cap_nouns)
                #cap_nouns_ids = self.tokenizer.convert_tokens_to_ids(cap_nouns_token)
                cap_pos_verbs_ids = self.tokenizer_v(cap_pos_verbs)
                cap_pos_nouns_ids = self.tokenizer_n(cap_pos_nouns)

                #cap_pos_verbs_token = self.tokenizer.tokenize(cap_pos_verbs)
                #cap_pos_verbs_ids = self.tokenizer.convert_tokens_to_ids(cap_pos_verbs_token)
                #cap_pos_nouns_token = self.tokenizer.tokenize(cap_pos_nouns)
                #cap_pos_nouns_ids = self.tokenizer.convert_tokens_to_ids(cap_pos_nouns_token)

                cap_neg_verbs_ids = self.tokenizer_v(cap_neg_verbs)
                cap_neg_nouns_ids = self.tokenizer_n(cap_neg_nouns)
                #cap_neg_verbs_token = self.tokenizer.tokenize(cap_neg_verbs)
                #cap_neg_verbs_ids = self.tokenizer.convert_tokens_to_ids(cap_neg_verbs_token)
                #cap_neg_nouns_token = self.tokenizer.tokenize(cap_neg_nouns)
                #cap_neg_nouns_ids = self.tokenizer.convert_tokens_to_ids(cap_neg_nouns_token)

                while len(cap_verbs_ids) < 3:
                    cap_verbs_ids.append(0)
                while len(cap_nouns_ids) <5:
                    cap_nouns_ids.append(0)
                while len(cap_pos_verbs_ids) < 3:
                    cap_pos_verbs_ids.append(0)
                while len(cap_pos_nouns_ids) < 5:
                    cap_pos_nouns_ids.append(0)
                while len(cap_neg_verbs_ids) < 3:
                    cap_neg_verbs_ids.append(0)
                while len(cap_neg_nouns_ids) < 5:
                    cap_neg_nouns_ids.append(0)
                cap_verbs_ids_[i] = np.array(cap_verbs_ids)
                cap_nouns_ids_[i] = np.array(cap_nouns_ids)
                cap_pos_verbs_ids_[i] = np.array(cap_pos_verbs_ids)
                cap_pos_nouns_ids_[i] = np.array(cap_pos_nouns_ids)
                cap_neg_verbs_ids_[i] = np.array(cap_neg_verbs_ids)
                cap_neg_nouns_ids_[i] = np.array(cap_neg_nouns_ids)

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids, pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
                cap_verbs_ids_,cap_nouns_ids_,cap_pos_verbs_ids_,cap_pos_nouns_ids_,cap_neg_verbs_ids_,cap_neg_nouns_ids_

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        #print(choice_video_ids)

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            #print(video_id)
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
        video_id, caption, caption_pos , caption_neg ,cap_verbs, cap_nouns, cap_pos_verbs, cap_pos_nouns, cap_neg_verbs, cap_neg_nouns= self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids, \
        pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
        cap_verbs_ids_, cap_nouns_ids_,cap_pos_verbs_ids_, cap_pos_nouns_ids_,cap_neg_verbs_ids_, cap_neg_nouns_ids_= self._get_text(video_id, caption, caption_pos , caption_neg,cap_verbs, cap_nouns, cap_pos_verbs, cap_pos_nouns, cap_neg_verbs, cap_neg_nouns)
        if self.video_feat_type=='Swin':
            video, video_mask, masked_video, video_labels_index = self._get_video_swin(choice_video_ids)
        elif self.video_feat_type=='Swin+Clip':
            video_swin, video_mask_swin, masked_video_swin, video_labels_index_swin = self._get_video_swin(choice_video_ids)
            video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)

        else:
            video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        #print(cap_verbs_ids, cap_nouns_ids,cap_pos_verbs_ids, cap_pos_nouns_ids,cap_neg_verbs_ids, cap_neg_nouns_ids)
        if self.video_feat_type == 'Swin+Clip':
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
                   pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                   video_swin, video_mask_swin, masked_video_swin, video_labels_index_swin,\
                   pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
                   cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_
        else:
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
                   pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
                   pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,pairs_decoder_mask_pos,pairs_decoder_mask_neg,pairs_input_caption_ids_pos,pairs_output_caption_ids_pos,pairs_input_caption_ids_neg,pairs_output_caption_ids_neg, \
                   cap_verbs_ids_, cap_nouns_ids_, cap_pos_verbs_ids_, cap_pos_nouns_ids_, cap_neg_verbs_ids_, cap_neg_nouns_ids_