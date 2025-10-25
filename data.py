import json 
import os
from dataclasses import asdict 
from typing import List,Union 

import clip 
import h5py
import numpy as np
import pandas as pd
import torch 
from torchvision.transforms import transforms
import webdataset as wds
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from img2dataset import download 
from torch import Tensor , nn
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from configs import DataDownloadConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def encode_text(label:Union[str,List[str]],model:nn.Module,device:str|torch.device)-> Tensor:
    text_tokens = clip.tokenize(label,truncate=True).to(device)
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()
@torch.no_grad()
def encode_image(img:Tensor,vae:AutoencoderKL)->Tensor:
    x = img.to(device).to(torch.float16)
    x = x*2 - 1
    encoded = vae.encode(x,return_dict=False)[0].sample()
    return encoded.cpu()
@torch.no_grad()
def decode_latents(out_latents:torch.Tensor,vae:AutoencoderKL)->Tensor:
    out  = vae.decode(out_latents.to(device))[0].cu()
    return ((out+1)/2).clip(0,1)

def quantize_latents(lat:Tensor,clip_val:float=20)->Tensor:
    "scale and quantize latents to uint8"
    lat_norm = lat.clip(-clip_val,clip_val) / clip_val
    return (((lat_norm + 1)/2)*255).to(torch.uint8)
def dequantize_latents(lat:Tensor,clip_val:float=20)->Tensor:
    lat_norm = (lat.to(torch.float16)/255)*2 - 1
    return lat_norm * clip_val
def append_to_dataset(dataset:h5py.dataset,new_data:Tensor)->None:
    new_size = dataset.shape[0] + new_data.shape[0]
    dataset.resize(new_size,axis=0)
    dataset[-new_data.shape[0]:] = new_data
def get_text_and_latent_embeddings_hdf5(
        dataloader:DataLoader,vae:AutoencoderKL,model:nn.Module , drive_save_path:str
)->None:
    img_latent_path = os.path.join(drive_save_path,"image_latents.hdf5")
    text_embed_path = os.path.join(drive_save_path,"text_encoding.hdf5")
    metadata_csv_path = os.path.join(drive_save_path,"metadata.csv")

    with h5py.File(img_latent_path,"a") as img_file , h5py.File(text_embed_path,"a") as text_file:
        if "image_latents" not in img_file:
            img_ds = img_file.create_dataset(
                "image_latents",
                shape = (0,4,32,32),
                maxshape = (None,4,32,32),
                dtype = "float16",
                chunks = True,
            )
        else:
            img_ds = img_file["image_latents"]
        if "text_encodings" not in text_file:
            text_ds = text_file.create_dataset(
                "text_encodings" ,shape=(0,768) , maxshape=(None,768),dtype="float16",chunks=True
            )
        else:
            text_ds = text_file["text_encodings"]
        for img,(label,url) in tqdm(dataloader):
            text_encoding = encode_text(label,model,device).cpu().numpy().astype(np.float16)
            img_encoding = encode_image(img,vae).cpu().numpy().astype(np.float16)

            append_to_dataset(img_ds,img_encoding)
            append_to_dataset(text_ds,text_encoding)
            metadata_df = pd.DataFrame({"text":label,"url":url})
            if os.path.exists(metadata_csv_path):
                metadata_df.to_csv(metadata_csv_path,mode="a",header=False,index=False)
            else:
                metadata_df.to_csv(metadata_csv_path,mode="w",header=True,index=False)

def download_and_process_data(
        latent_save_path:str,
        raw_imgs_save_path:str="raw_imgs",
        csv_path:str="imgs.csv",
        image_size:int=64,
        bs=1,
        cation_col:str="caption",
        url_col:str="url",
        download_data:bool=True,
        number_sample_per_shard:int=10000,
):
    if not os.path.exists(raw_imgs_save_path):
        os.mkdir(raw_imgs_save_path)
    if not os.path.exists(latent_save_path):
        os.mkdir(latent_save_path)
    if download_data:
        download(
            processes_count=8,
            thread_count=16,
            url_list=csv_path,
            output_folder=raw_imgs_save_path,
            output_format="webdataset",
            input_format="csv",
            url_col=url_col,
            caption_col=cation_col, 
            enable_wandb=False,
            number_sample_per_shard=number_sample_per_shard,
            distributor="multiprocessing",
            rezie_mode="centre_crop"
        )
        files = os.listdir(raw_imgs_save_path)
        tar_files = [os.path.join(raw_imgs_save_path,file) for file in files if file.endswith(".tar")]
        print(tar_files)
        dataset = wds.WebDataset(tar_files)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
