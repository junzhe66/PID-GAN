# PID-GAN: Physics-Informed Discriminator Generative Adversarial Network for Nowcasting Extreme Rainfall

## Overview
PID-GAN is a specialized Generative Adversarial Network designed to enhance the accuracy and realism of rainfall nowcasting predictions. This model incorporates physical laws into the adversarial training process, enabling it to generate outputs that closely mirror real-world meteorological conditions. The README offers detailed instructions for setting up, configuring, and training the PID-GAN model, making it an invaluable tool for improving short-term rainfall prediction and related meteorological applications.

## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Modifying Physical Parameters](#modifying-physical-parameters)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)
- [Contact](#contact)

## Installation
Instructions for installing the PID-GAN model and any required libraries or dependencies.

git clone https://github.com/junzhe66/PID-GAN.git
cd PID-GAN
pip install -r requirements.txt

##
## Launch a Training Run

Initiate the model training process by executing the following command:

python src/main.py
Logs and training metrics are automatically synchronized with Weights & Biases. To disable this feature, set wandb.mode=disabled.

## Configuration
The configuration of PID-GAN is managed through YAML files located in the config/ directory, with config/trainer.yaml serves as the primary configuration file. Direct modifications to these files are the simplest method to adjust settings such as learning rates, batch sizes, and other model parameters.

For advanced configuration options and dynamic parameter adjustments, PID-GAN utilizes Hydra. Refer to the Hydra documentation for comprehensive guidance on configuration management.


## Adjusting Physical Parameters
Physical parameters and equations are defined in collector.py and Phy.py. Modify these files to change the physical constraints or equations the model adheres to during training.
