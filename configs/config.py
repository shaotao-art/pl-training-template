device = 'cuda'

num_ep = 100
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-4,
        # momentum=0.9,
    )
)

lr_sche_config = dict(
    type = 'cosine',
    config = dict(
        # epoches=[60, 80],
        # muls=[0.1, 0.1]
    )
)

act_type = 'relu'
norm_type = 'none'
model_config = dict(
    channels=[32, 64, 128, 256, 256],
    num_block_per_stage=2,
    block_type='TwoConvBlock',
    act_type=act_type,
    norm_type=norm_type,
    # num_channels_per_gn_group = 1,
    base_block_config=dict(
        in_channels=None, 
        out_channels=None,
        # reduction=16,
        act=act_type,
        norm_config = dict(
            type=norm_type,
            config=dict(
            )
        )
    )
)



cifar_data_root = 'DATA'
train_data_config = dict(
    dataset_config = dict(
        root = cifar_data_root,
    ), 
    transform_config = dict(
        img_size=32,
        normalize_config=dict(
            mean=(0.5, ),
            std=(0.5, )
        )
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
    transform_config = dict(
        img_size=32,
        normalize_config=dict(
            mean=(0.5, ),
            std=(0.5, )
        )
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