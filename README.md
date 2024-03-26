
# PID-GAN

PID-GAN leverages advanced generative adversarial network principles, incorporating physically informed design to optimize performance and realism in generated outputs. This README guide outlines how to initiate training runs and configure the model to fit specific requirements.
Related project report: https://repository.tudelft.nl/islandora/object/uuid%3A4fe7cd89-5f7c-468b-8f8b-d2c493be9386
## Launch a Training Run

Initiate the model training process by executing the following command:

python src/main.py
Logs and training metrics are automatically synchronized with Weights & Biases. To disable this feature, set wandb.mode=disabled.

## Configuration
The configuration of PID-GAN is managed through YAML files located in the config/ directory, with config/trainer.yaml serves as the primary configuration file. Direct modifications to these files are the simplest method to adjust settings such as learning rates, batch sizes, and other model parameters.

For advanced configuration options and dynamic parameter adjustments, PID-GAN utilizes Hydra. Refer to the Hydra documentation for comprehensive guidance on configuration management.


## Adjusting Physical Parameters
Physical parameters and equations are defined in collector.py and Phy.py. Modify these files to change the physical constraints or equations the model adheres to during training.
