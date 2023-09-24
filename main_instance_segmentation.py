import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra 
from dotenv import load_dotenv
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model, 
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf,open_dict
from tqdm import tqdm
import json 
import torch



 
####################################################################################################################################
import numpy as np
from datasets.scannet200.scannet200_constants import CLASS_LABELS_200
from datasets.scannet200.scannet200_splits import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
####################################################################################################################################
############################
CLASS_LABELS_200 = list(CLASS_LABELS_200)
CLASS_LABELS_200.remove('floor')
CLASS_LABELS_200.remove('wall')
# HEAD_CATS_SCANNET_200.remove('floor')
# HEAD_CATS_SCANNET_200.remove('wall')
MAP_STRING_TO_ID = {CLASS_LABELS_200[i] : i for i in range(len(CLASS_LABELS_200))}
MAP_STRING_TO_ID['background'] = 253
############################

def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration 
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def train(cfg: DictConfig):
    if cfg.general.OW_task != "task1" and cfg.general.finetune:
        cfg.general.save_dir = cfg.general.save_dir+"_finetune"
        cfg.general.logg_suffix = cfg.general.logg_suffix+"_finetune"
        
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = [] 
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())
    # callbacks = [callbacks[0],callbacks[2]] #callbacks{1}: lr
    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    # resume_from = "/l/users/mohamed.boudjoghra/Research/Mask3D/alvis_exp/multiscale_10_AL_CC_unkn_dist_top20/task1/epoch=549-val_mean_ap_50=0.527.ckpt"
    # checkpoint = torch.load(resume_from)
    # model.load_state_dict(checkpoint["state_dict"])
    if not cfg.general.train_oracle and cfg.general.train_mode:
        if cfg.general.OW_task != "task1" and (not cfg.general.finetune):
            task = cfg.general.save_dir.split('/')[-1]
            prev_task = task.replace(task[-1], str(int(task[-1])-1))
            if os.path.exists(cfg.general.save_dir.replace(task, prev_task+"_finetune")):
                list_dir = os.listdir(cfg.general.save_dir.replace(task, prev_task+"_finetune"))
                for file in list_dir:
                    file_s = file.split("_")
                    if "ap" in file_s:
                        path_to_prev_task = cfg.general.save_dir.replace(task, prev_task+"_finetune")+"/"+file
                        break
                    else:
                        path_to_prev_task = cfg.general.save_dir.replace(task, prev_task)+"/"+"last-epoch.ckpt"
            else:
                path_to_prev_task = cfg.general.save_dir.replace(task, prev_task)+"/"+"last-epoch.ckpt"   
                  
            checkpoint = torch.load(path_to_prev_task)
            if not os.path.exists(cfg.general.save_dir+"/"+"last-epoch.ckpt"):
                model.load_state_dict(checkpoint["state_dict"]) 
        elif cfg.general.OW_task != "task1" and cfg.general.finetune:
            resume_from = os.path.join(cfg.general.save_dir.replace("_finetune",""),"last.ckpt")
            checkpoint = torch.load(resume_from)
            if not os.path.exists(cfg.general.save_dir+"/"+"last.ckpt"):
                model.load_state_dict(checkpoint["state_dict"]) 
    runner.fit(model)
    
@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reasonimage.png
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    runner.test(model)

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    if cfg['general']['train_mode']:
        train(cfg)
    else:
        test(cfg)
        

if __name__ == "__main__":    main()
 