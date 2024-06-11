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