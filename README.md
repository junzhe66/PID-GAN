
# PID-GAN

how to train the model

run the command: python src/main.py 

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

The physical parameters are loaded by collector.py and calculate the corresponding equation in Phy.py.



## Configuration
- All model parameters can be changed in the folder config. 
- All training parameters of the models can be changed in `config/trainer.yaml`.
- The simplest way to customize the configuration is to edit these files directly.

