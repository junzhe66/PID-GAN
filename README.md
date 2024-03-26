
# PID-GAN

1, how to train the model: run the command: python src/main.py 

2, By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

3, The physical parameters are loaded by collector.py and the corresponding equation in Phy.py.

4,  All model parameters can be changed in the folder config. 
5,  All training parameters of the models can be changed in `config/trainer.yaml`.
6,  The simplest way to customize the configuration is to edit these files directly.

