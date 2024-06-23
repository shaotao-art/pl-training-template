import transformers
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime
from typing import List
import math
from functools import partial

def get_time_str():
    currentDateAndTime = datetime.now()
    day = currentDateAndTime.strftime("%D").replace('/', '-')
    time = currentDateAndTime.strftime("%H:%M:%S")
    currentTime = day + '/' + time
    return currentTime

def get_callbacks(ckp_config):
    checkpoint_callback = ModelCheckpoint(**ckp_config)
    callbacks = []
    callbacks.append(LearningRateMonitor('step'))
    callbacks.append(checkpoint_callback)
    return callbacks


def get_step_lr_sche(optimizer, 
                     epoches: List[int],
                     muls: List[float]):
    """at epoches, mul lr with muls"""
    def step_lr_fn(epoch, epoches: List[int], muls: List[float]):
        if len(epoches) == 1:
            if epoch < epoches[0]:
                return 1.0
            else:
                return muls[0]
        elif len(epoches) == 2:
            if epoch < epoches[0]:
                return 1.0
            elif epoch < epoches[1]:
                return muls[0]
            else:
                return muls[0] * muls[1]
        else:
            raise NotImplementedError
    
    fn = partial(step_lr_fn, epoches=epoches, muls=muls)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    return scheduler    

def get_cool_down_lr_sche(optimizer,
                        total_steps, 
                        warm_up_steps, 
                        cool_down_steps,
                        cool_down_schedule = 'inver_sqrt'):
    def cool_down_lr_fn(step, 
                        total_steps, 
                        warm_up_steps, 
                        cool_down_steps, 
                        cool_down_schedule):
        if step < warm_up_steps:
            # linear warm up
            return ((step + 1)/ warm_up_steps)
        elif step < total_steps - cool_down_steps:
            # constant lr
            return 1.0
        else:
            if cool_down_schedule == 'linear':
                # linear cool down
                return 1.0 - (step - (total_steps - cool_down_steps)) / cool_down_steps
            elif cool_down_schedule == 'inver_sqrt':
                # inverse sqrt cool down
                return 1.0 - math.sqrt((step - (total_steps - cool_down_steps)) / cool_down_steps)
            elif cool_down_schedule == 'cosine':
                # cosine cool down
                return  0.5 * (1 + math.cos((step - (total_steps - cool_down_steps)) / cool_down_steps * math.pi))
            else:
                raise NotImplementedError
        
    fn = partial(cool_down_lr_fn, total_steps=total_steps, warm_up_steps=warm_up_steps, cool_down_steps=cool_down_steps, cool_down_schedule=cool_down_schedule)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    return scheduler



def get_opt_lr_sch(optimizer_config, lr_sche_config, model):
    if optimizer_config.type == 'sgd':
        optimizer = optim.SGD
    elif optimizer_config.type == 'adam':
        optimizer = optim.Adam
    elif optimizer_config.type == 'adamw':
        optimizer = optim.AdamW
    else:
        raise NotImplementedError
    
    no_decay_gps = []
    decay_gps = []
    for k, v in model.named_parameters():
        if v.ndim <= 1:
            no_decay_gps.append(v)
        else:
            decay_gps.append(v)
    optimizer = optimizer([{'params': no_decay_gps, 'weight_decay': 0.0},
                             {'params': decay_gps}],
                             **optimizer_config.config)
    if lr_sche_config.type == 'constant':
        lr_sche = transformers.get_constant_schedule(optimizer)
    elif lr_sche_config.type == 'linear':
        lr_sche = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                **lr_sche_config.config)
    elif lr_sche_config.type == 'cosine':
        lr_sche = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                **lr_sche_config.config)
    elif lr_sche_config.type == 'cool_down':
        lr_sche = get_cool_down_lr_sche(optimizer=optimizer,
                                        **lr_sche_config.config)
        
    elif lr_sche_config.type == 'step':
        lr_sche = get_step_lr_sche(optimizer=optimizer,
                                   **lr_sche_config.config)   
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': lr_sche,
            'interval': 'epoch'
        }
    }
    elif lr_sche_config.type == 'reduce':
        lr_sche = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        **lr_sche_config.config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_sche,
                'monitor': lr_sche_config.config.reduce_monitor,
                'interval': 'step'
            }
        }
    else:
        raise NotImplementedError
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': lr_sche,
            'interval': 'step'
        }
    }
     
