import copy 
from dataclasses import asdict

import numpy as np 
import torch
import torchvision
import torchvision.utils as vutils 
import wandb 
from accelerate import Accelerator
from diffusers import AutoencoderKL 
from PIL.Image import Image
from torch import Tensor , nn 
from torch.utils.data import DataLoader , TensorDataset
from tqdm import tqdm 

from denoiser import Denoiser
from diffusion import DiffusionGenerator
from configs import ModelConfig   

to_pil = torchvision.transforms.ToPILImage()

def eval_gen(diffuser:DiffusionGenerator,labels:Tensor,img_size:int)->Image:
    class_guidance = 4.5
    seed = 10
    out , _ = diffuser.generate(
        labels=torch.repeat_interleave(labels,2,dim=0),
        num_imgs = 16,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f = 0.1,
        img_size = img_size
    )
    out = to_pil((vutils.make_grid((out+1)/2,nrow=8,padding=4)).float().clip(0,1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")
    return out
def count_parameters(model:nn.Module)->int:
    return sum(p.enumel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_layer(model:nn.Module):
    for name,param in model.named_parameters():
        print(f"{name}:{param.numel()} parameters")


def update_ema(ema_model:nn.Module,model:nn.Module,alpha:float=0.999):
    with torch.no_grad():
        for ema_param ,model_param in zip(ema_model.parameters(),model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data,alpha=1 - alpha)


def main(config:ModelConfig)->None:
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config

    log_with = "wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16",log_with=log_with)

    accelerator.print("Loading Data:")
    latent_train_data = torch.tensor(np.load(dataconfig.latent_path),dtype=torch.float32)
    train_label_embeddings = torch.tensor(np.load(dataconfig.val_path),dtype=torch.float32)
    emb_val = torch.tensor(np.load(dataconfig.val_path),dtype=torch.float32)
    dataset = TensorDataset(latent_train_data,train_label_embeddings)
    train_loader = DataLoader(dataset,batch_size = train_config.batch_size,shuffle=True)
    vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name,torch_dtype=config.vae_cfg.vae_dtype)

    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)
    
    model = Denoiser(**asdict(denoiser_config)) 

    loss_fn  = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(),lr=train_config.lr)

    if not train_config.from_scratch:
        accelerator.print("Loading Model")
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0 
    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model,vae,accelerator.device,torch.float32)
    accelerator.print("model prep")
    model,optimizer,train_loader = accelerator.prepare(model,optimizer,train_loader)

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    for i in range(1,train_config.n_epoch + 1):
        accelerator.print(f"epoch{i}")

        for x,y in tqdm(train_loader):
            x = x/config.vae_cfg.scaling_factor
            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a,train_config.beta_b,len(x)),device = accelerator.device
            )
            signal_level = 1-noise_level
            noise = torch.randn_like(x)
            x_noisy = noise_level.view(-1,1,1,1) * noise + signal_level.view(-1,1,1,1)*x
            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y 

            prob = 0.15 
            mask = torch.rand(y.size(0),device=accelerator.device) < prob 
            label[mask] = 0 

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    out = eval_gen(diffuser,emb_val,img_size=denoiser_config.img_size)
                    out.save(f"img.jpg")
                    
                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        "model_ema":ema_model.state_dict(),
                        "opt_state":opt_unwrapped.state_dict(),
                        "global_step":global_step
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict,train_config.model_name)
            model.train()

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                pred = model(x_noisy,noise_level.view(-1,1),label)
                loss = loss_fn(pred,x)
                accelerator.log({"train_loss":loss.item()},step=global_step)
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model,model,alpha=train_config.alpha)
            global_step += 1
    accelerator.end_training()

