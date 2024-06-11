device = 'cuda'

num_ep = 100
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-4
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # warm_up_epoch=0
    )
)

img_size = 32
patch_size = 4
# follow vit-base
model_config = dict(
    img_size = img_size,
    patch_size = patch_size,
    torch_transformer_encoder_config = dict(
        num_layers=12,
        layer_config = dict(
            d_model=128,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            nhead=4,
            norm_first=True,
            batch_first=True,
            bias=True
        ))
)



cifar_data_root = 'DATA'
train_data_config = dict(
    dataset_config = dict(
        root = cifar_data_root,
    ), 
    data_loader_config = dict(
        batch_size = 64,
        num_workers = 4,
    )
)
test_data_config = dict(
    dataset_config = dict(
        root = cifar_data_root,
    ), 
    data_loader_config = dict(
        batch_size = 64,
        num_workers = 4,
        
    )
)



resume_ckpt_path = None
load_weight_from = None

# ckp
ckp_config = dict(
   save_last=None, 
   every_n_epochs=None,
#    monitor='val_mae',
#    mode='min',
#    filename='{epoch}-{val_mae:.3f}'
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    # val_check_interval=0.5, # val after k training batch 0.0-1.0, or a int
    check_val_every_n_epoch=1
)


# LOGGING
enable_wandb = True
wandb_config = dict(
    project = 'backbone-exp',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'