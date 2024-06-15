device = 'cuda'

num_ep = 100
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 1e-4,
        # momentum=0.9,
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # epoches=[60, 80],
        # muls=[0.1, 0.1]
    )
)

model_name = ""
model_config = dict(
    model_name = model_name,
    num_classes=6
)


                
cifar_data_root = 'DATA'
train_data_config = dict(
    dataset_config = dict(
        csv_file='data/train.csv', 
        id_column='essay_id', 
        text_column='full_text',
        label_column='score',
        mode='train'
    ), 
    tokenizer_name =model_name,

    data_loader_config = dict(
        batch_size = 16,
        num_workers = 4,
    )
)
test_data_config = dict(
    dataset_config = dict(
        csv_file='data/train.csv', 
        id_column='essay_id', 
        text_column='full_text',
        label_column='score',
        mode='val'
    ), 
    tokenizer_name =model_name,
    data_loader_config = dict(
        batch_size = 16,
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
    project = 'nlp-cls',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'
