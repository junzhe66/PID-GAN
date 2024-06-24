# PID-GAN: Physics-Informed Discriminator Generative Adversarial Network for Nowcasting Extreme Rainfall

## Overview
PID-GAN is a specialized Generative Adversarial Network designed to enhance the accuracy and realism of rainfall nowcasting predictions. This model incorporates physical laws into the adversarial training process, enabling it to generate outputs that closely mirror real-world meteorological conditions. The README offers detailed instructions for setting up, configuring, and training the PID-GAN model, making it an invaluable tool for improving short-term rainfall prediction and related meteorological applications. https://arxiv.org/abs/2406.10108 

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
    Install all necessary Python libraries listed in the all_packages.txt file. Ensure you have Python and pip installed on your machine

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
Configuration of VQ-GAN can be modified through YAML files located in the `config/` directory. The `config/tokenizer/default.yaml` file allows you to adjust settings specific to the tokenizer. For the transformer configuration, modifications can be made in the `config/world_model/default.yaml` file. The `config/trainer.yaml` file manages the training configuration for the entire model, including settings like learning rates and batch sizes.

## Training
Launch the model training using the command specified in Getting Started. Monitor training progress through the command line or via Weights & Biases integration.

## Dataset

Physical parameters and equations are defined in `collector.py` and `Phy.py`. Modify these files to change the physical constraints or equations the model adheres to during training. Specifically, the calculation of Equation 9 from the paper is implemented in `Phy.py`. The model's accuracy and realism in rainfall nowcasting rely significantly on the quality and characteristics of the input data.

For real-world application and enhancement of the model's performance, we utilize the "Archived 5-min rainfall accumulations from a radar dataset for the Netherlands". This dataset provides high-resolution rainfall data, which is crucial for refining and testing our model's predictive capabilities under various meteorological conditions.

1. **Dataset available here**: [Archived 5-min rainfall accumulations for the Netherlands](https://data.4tu.nl/articles/dataset/Archived_5-min_rainfall_accumulations_from_a_radar_dataset_for_the_Netherlands/12675278)
2. **ERA5 dataset**: [ERA5 Reanalysis Dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview)
3. **AWS dataset**: [KNMI Hourly Data](https://www.daggegevens.knmi.nl/klimatologie/uurgegevens)

The analysis of the physical data to perform the cubic interpolation and kriging interpolation, as well as how to generate the interpolated dataset, can be found in the files of the `PHY` folder. 

You can change the path of where you want to save the generated dataset in the function `save_temp_to_hdf5`. The radar dataset selection and analysis can be found in this repository: [Nowcasting of Extreme Precipitation](https://github.com/bbbbihr/Nowcasting-of-extreme-precipitation).

## DataLoader


