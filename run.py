from mmengine import Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import os
import math
import argparse

import torch
from torch import nn
import torch.nn.functional as F



from run_utils import get_callbacks, get_time_str, get_opt_lr_sch
from cifar_10_dataset import get_train_data, get_val_data
from models.cnn_model import CnnModel
from models.vit import TorchVit, TorchConvVit
from cv_common_utils import show_or_save_batch_img_tensor


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        ########## ================ MODEL ==================== ##############
        if config.model_config.get('patch_size', None) is not None:
            if config.model_config.get('conv_channels', None) is None:
                self.model = TorchVit(**config.model_config)
            else: 
                self.model = TorchConvVit(**config.model_config)
        else:
            self.model = CnnModel(**config.model_config)
        self.loss_fn = nn.CrossEntropyLoss()
        ########## ================ MODEL ==================== ##############
                

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        pred = self.model(imgs)
        loss = self.loss_fn(pred, labels)
        with torch.no_grad():
            pred_cls = torch.argmax(pred, dim=1)
            train_acc = (pred_cls == labels).float().mean()
        self.log_dict({'train_loss': loss,
                       'train_acc': train_acc})
        
        if self.current_epoch == 0 and batch_idx == 0:
            b_s = imgs.shape[0]
            vis_img = show_or_save_batch_img_tensor(imgs, int(math.sqrt(b_s)), denorm=True, mode='return')
            self.logger.experiment.add_image(tag=f'train_batch', 
                                img_tensor=vis_img, 
                                global_step=self.global_step,
                                dataformats='HWC',
                                )
        return loss
    
    
    def on_validation_epoch_start(self) -> None:
        self.num_right = 0
        self.total_num = 0
    
    def on_validation_epoch_end(self) -> None:
        self.log_dict({'val_acc': self.num_right / self.total_num})
        
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        imgs, labels = batch
        pred = self.model(imgs)
        pred_cls = torch.argmax(pred, dim=1)
        self.num_right += torch.sum(pred_cls == labels).item()
        self.total_num += imgs.shape[0]
        
        if self.current_epoch == 0 and batch_idx == 0:
            b_s = imgs.shape[0]
            vis_img = show_or_save_batch_img_tensor(imgs, int(math.sqrt(b_s)), denorm=True, mode='return')
            self.logger.experiment.add_image(tag=f'val_batch', 
                                img_tensor=vis_img, 
                                global_step=self.global_step,
                                dataformats='HWC',
                                )
        

    def configure_optimizers(self):
        return get_opt_lr_sch(self.config.optimizer_config, 
                              self.config.lr_sche_config,  
                              self.model)
    




def run(args):
    config = Config.fromfile(args.config)
    config = modify_config(config, args)
    
    # make ckp accord to time
    time_str = get_time_str()
    config.ckp_root = '-'.join([time_str, config.ckp_root, f'[{args.run_name}]'])
    config.ckp_config['dirpath'] = config.ckp_root
    os.makedirs(config.ckp_root, exist_ok=True)
    config.run_name = args.run_name
    # logger
    
    # wandb_logger = None
    # if config.enable_wandb:
    #     wandb_logger = WandbLogger(**config.wandb_config,
    #                             name=args.wandb_run_name)
    #     wandb_logger.log_hyperparams(config)
    logger = TensorBoardLogger(save_dir=config.ckp_root,
                               name=config.run_name)
    
    # DATA
    print('getting data...')
    train_data, train_loader = get_train_data(config.train_data_config)
    val_data, val_loader = get_val_data(config.test_data_config)
    print(f'len train_data: {len(train_data)}, len val_loader: {len(train_loader)}.')
    print(f'len val_data: {len(val_data)}, len val_loader: {len(val_loader)}.')
    print('done.')


    # lr sche 
    if config.lr_sche_config.type in ['linear', 'cosine']:
        if config.lr_sche_config.config.get('warm_up_epoch', None) is not None:
            warm_up_epoch = config.lr_sche_config.config.warm_up_epoch
            config.lr_sche_config.config.pop('warm_up_epoch')
            config.lr_sche_config.config['num_warmup_steps'] = int(warm_up_epoch * len(train_loader))
        else:
            config.lr_sche_config.config['num_warmup_steps'] = 0
        config.lr_sche_config.config['num_training_steps'] = config.num_ep * len(train_loader)
    
    # MODEL
    print('getting model...')
    model = Model(config)
    print(model)
    if 'load_weight_from' in config and config.load_weight_from is not None:
        # only load weights
        state_dict = torch.load(config.load_weight_from)['state_dict']
        model.load_state_dict(state_dict)
        print(f'loading weight from {config.load_weight_from}')
    print('done.')
    
    
    callbacks = get_callbacks(config.ckp_config)
    config.dump(os.path.join(config.ckp_root, 'config.py'))
    

    #TRAINING
    print('staring training...')
    if args.find_lr:
        max_steps = args.max_steps
    else:
        max_steps = -1
    resume_ckpt_path = config.resume_ckpt_path if 'resume_ckpt_path' in config else None
    trainer = pl.Trainer(accelerator=config.device,
                         max_epochs=config.num_ep,
                         callbacks=callbacks,
                         logger=logger,
                         enable_progress_bar=True,
                         max_steps=max_steps,
                        #  gradient_clip_val=1.0,
                         **config.trainer_config
                         )
    
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_ckpt_path
                )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to mmcv config file")
    parser.add_argument("--run_name", required=True, type=str, help="wandb run name")
    
    parser.add_argument("--find_lr", action='store_true', help="whether to find learning rate")
    parser.add_argument("--max_steps", type=int, default=-100, help='max step to run when find lr')

    # common args to overwrite config
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    args = parser.parse_args()
    return args


def modify_config(config, args):
    if args.lr is not None:
        config['optimizer_config']['config']['lr'] = args.lr
    if args.wd is not None:
        config['optimizer_config']['config']['weight_decay'] = args.wd
    return config

if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(42)
    run(args)