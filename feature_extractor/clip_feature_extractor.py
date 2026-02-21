import argparse
import os
from tqdm import tqdm
import torch

import pickle
import pathlib
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip

from dataloaders.dataloader_msvd_raw import MSVD_Raw_DataLoader
from dataloaders.dataloader_msrvtt_raw import MSRVTT_Raw_DataLoader
from dataloaders.dataloader_zyh_raw import ZYH_Raw_DataLoader

# Argument
class args:
    msvd = True # or msvd = False for MSR-VTT
    max_frames = 20
    pretrined_clip4clip_dir='pretrained'
    
def get_args():
    parser = argparse.ArgumentParser(description="CLIP Feature Extractor")
    parser.add_argument('--dataset_type', choices=['msvd', 'msrvtt','zyh'], default='msvd', type=str, help='msvd or msrvtt')
    parser.add_argument('--dataset_dir', type=str, default='../dataset', help='should be pointed to the location where the MSVD and MSRVTT dataset located')
    parser.add_argument('--save_dir', type=str, default='../extracted_feats', help='location of the extracted features')
    parser.add_argument('--slice_framepos', choices=[0,1,2], type=int, default=2,
                        help='0: sample from the first frames; 1: sample from the last frames; 2: sample uniformly.')
    parser.add_argument('--max_frames', type=int, default=20, help='max sampled frames')
    parser.add_argument('--frame_order', type=int, choices=[0,1,2], default=0, help='0: normal order; 1: reverse order; 2: random order.')
    parser.add_argument('--pretrained_clip4clip_dir', type=str, default='pretrained_clip4clip/', help='path to the pretrained CLIP4Clip model') 
    parser.add_argument('--device', choices=["cpu", "cuda"], type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--pretrained_clip_name', type=str, choices=["ViT-B/32", "ViT-B/16"], default="ViT-B/32")
    
    args = parser.parse_args()
    
    if args.device == "cuda":
        args.device = torch.device('cuda')
    
    if args.dataset_type=="msvd":
        dset_path = os.path.join(args.dataset_dir,'MSVD')
        args.videos_path = os.path.join(dset_path,'raw') # video .avi    

        args.data_path =os.path.join(os.path.join(dset_path,'captions','youtube_mapping.txt'))
        args.max_words = 30
        
        save_dir = os.path.join(args.save_dir, "msvd")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        args.save_file = os.path.join(save_dir,'MSVD_CLIP4Clip_features.pickle')
        
        args.pretrained_clip4clip_path = os.path.join(args.pretrained_clip4clip_dir, 'msvd','pytorch_model.bin')

    elif args.dataset_type=="msrvtt":
        dset_path = os.path.join(args.dataset_dir,'MSRVTT')
        args.videos_path = os.path.join(dset_path,'raw') 
        
        args.data_path=os.path.join(dset_path,'MSRVTT_data.json')
        args.max_words = 73
        args.csv_path = os.path.join(dset_path,'msrvtt.csv')

        save_dir = os.path.join(args.save_dir, "msrvtt")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        args.save_file = os.path.join(save_dir,'MSRVTT_CLIP4Clip_features.pickle')
        
        args.pretrained_clip4clip_path = os.path.join(args.pretrained_clip4clip_dir, 'msrvtt','pytorch_model.bin')

    elif args.dataset_type == "zyh":
        args.videos_path =args.dataset_dir
        args.max_words = 30

        save_dir = os.path.join(args.save_dir, "zyh")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        args.save_file = os.path.join(save_dir, 'ZYH_MSRVTT_CLIP4Clip_features.pickle')

        args.pretrained_clip4clip_path = os.path.join(args.pretrained_clip4clip_dir, 'msrvtt', 'pytorch_model.bin')
    return args
    
def get_dataloader(args):
    
    dataloader = None
    if args.dataset_type=="msvd":
        dataloader = MSVD_Raw_DataLoader(
            data_path=args.data_path,
            videos_path=args.videos_path,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
            transform_type = 0,
        ) 
    elif args.dataset_type=="msrvtt":
        dataloader = MSRVTT_Raw_DataLoader(
            csv_path=args.csv_path,
            videos_path=args.videos_path,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
            transform_type = 0,
        )
    elif args.dataset_type=="zyh":
        dataloader = ZYH_Raw_DataLoader(
            videos_path=args.videos_path ,
            max_frames=args.max_frames,
            frame_order=args.frame_order,
            slice_framepos=args.slice_framepos,
            transform_type = 0,
        )
    return dataloader
    
def load_model(args):
    model_state_dict = torch.load(args.pretrained_clip4clip_path, map_location='cpu')
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed') ## 定义缓存目录
    ## 从预训练模型的状态字典中创建模型
    model = CLIP4Clip.from_pretrained('cross-base', cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    clip = model.clip.to(args.device)
    return clip

def main():
    args = get_args()
    dataloader = get_dataloader(args)
    model = load_model(args)
    model.eval()

    with torch.no_grad():
        data ={}
        stop = False
        with open(args.save_file, 'wb') as handle:## 打开一个文件用于保存结果

            for i in tqdm(range(len(dataloader))):
                video_id,video,video_mask = dataloader[i]## 获取视频ID，视频和视频掩码
                tensor = video[0]
                tensor = tensor[video_mask[0]==1,:]
                tensor = torch.as_tensor(tensor).float()
                video_frame,num,channel,h,w = tensor.shape
                tensor = tensor.view(video_frame*num, channel, h, w)

                video_frame,channel,h,w = tensor.shape
                ## 使用模型对视频进行编码
                output = model.encode_image(tensor.to(args.device), video_frame=video_frame).float().to(args.device)
                output = output.detach().cpu().numpy()
                data[video_id]=output # # 将输出保存到字典中
                del output
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)## 将字典保存到文件中

if __name__ == "__main__":
    main()
# python clip_feature_extractor.py --dataset_type msrvtt --save_dir ../extracted_feats --dataset_dir ../dataset

# python clip_feature_extractor.py --dataset_type msvd --save_dir ../extracted_feats --dataset_dir ../dataset



# python clip_feature_extractor.py --dataset_type zyh --save_dir ../extracted_feats_zyh --dataset_dir /media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/Clip4Caption_my/feature_extractor/datasets/zyh/raw