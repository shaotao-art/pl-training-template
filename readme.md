# Pytorch Lightning Training Template
## Features
* tensorboard logger
* lr sche: constant, linear, cosine...
* lr finder
* config file 
* easy to overwrite config file in command line

## Guideline
* `configs/config.py` to modify config
* `run.py` to see main pipeline
* `run_utils.py` to see lr_sche, callbacks

## Commands
`run.sh` to launch a single run
`run_lr_find.sh` to launch a learning rate find run
* remember to channge configs in .sh file