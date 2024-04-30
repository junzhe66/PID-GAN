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
   ```
## Configuration
Configuration of PID-GAN is managed through YAML files located in the config/ directory. The config/trainer.yaml serves as the primary configuration file. Modifications here can adjust settings like learning rates, batch sizes, and other model parameters.

For dynamic parameter adjustments and advanced configuration options, PID-GAN utilizes Hydra. Comprehensive guidance on configuration management can be found in the Hydra documentation.

## Training
Launch the model training using the command specified in Getting Started. Monitor training progress through the command line or via Weights & Biases integration.

## Adjusting Physical Parameters

Physical parameters and equations are defined in `collector.py` and `Phy.py`. Modify these files to change the physical constraints or equations the model adheres to during training. The model's accuracy and realism in rainfall nowcasting rely significantly on the quality and characteristics of the input data.

For real-world application and enhancement of the model's performance, we utilize the "Archived 5-min rainfall accumulations from a radar dataset for the Netherlands". This dataset provides high-resolution rainfall data, which is crucial for refining and testing our model's predictive capabilities under various meteorological conditions.

Dataset available here: [Archived 5-min rainfall accumulations for the Netherlands](https://data.4tu.nl/articles/dataset/Archived_5-min_rainfall_accumulations_from_a_radar_dataset_for_the_Netherlands/12675278)

Ensure that any adjustments to physical parameters or datasets are tested thoroughly to maintain the integrity and accuracy of the model outputs.




