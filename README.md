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

To set up the PID-GAN model on your system, follow these steps:

1. **Clone the Repository:**
   Use Git to clone the repository to your local machine. Open your terminal and run the following command:
   ```bash
   git clone https://github.com/junzhe66/PID-GAN.git

2. **Navigate to the Project Directory:**
   Change into the project directory using:
   ```bash
   cd PID-GAN

3. **Install Required Dependencies:**
   Install all necessary Python libraries from the requirements.txt file. Ensure you have Python and pip installed on your machine, then run:
      ```bash
   pip install -r requirements.txt

## Getting Started
   **To initiate a training run of the PID-GAN model, use the following command:**
   ```bash
   python src/main.py 
   ```
   **Logs and training metrics are synchronized with Weights & Biases by default. To disable this feature:**
    ```bash
set wandb.mode=disabled

## Configuration

Initiate the model training process by executing the following command:

python src/main.py
Logs and training metrics are automatically synchronized with Weights & Biases. To disable this feature, set wandb.mode=disabled.

## Configuration
The configuration of PID-GAN is managed through YAML files located in the config/ directory, with config/trainer.yaml serves as the primary configuration file. Direct modifications to these files are the simplest method to adjust settings such as learning rates, batch sizes, and other model parameters.

For advanced configuration options and dynamic parameter adjustments, PID-GAN utilizes Hydra. Refer to the Hydra documentation for comprehensive guidance on configuration management.


## Adjusting Physical Parameters
Physical parameters and equations are defined in collector.py and Phy.py. Modify these files to change the physical constraints or equations the model adheres to during training.
