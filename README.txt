# Geothermal Heat Flow Mapping of Germany

This repository provides data and code for modeling and mapping geothermal heat flow across Germany. The project aims to integrate multiple geophysical and geological data sources, offering insights with uncertainty quantification.

## Authors

- **Magued Al-Aghbary**  
  *Email:* mauged.alaghbary@outlook.com
- **Mohamed Sobh**  
  *Email:* Mohamed.Sobh@leibniz-liag.de

## Installation Instructions

### Prerequisites

- Install [Anaconda](https://docs.anaconda.com/anaconda/install/) (latest version).
- Install [GMT (Generic Mapping Tools)](https://www.generic-mapping-tools.org/download/).

### Setup

1. Create the environment:

   ```bash
   conda env create -n GQ -f requirements.yml
   ```

2. Activate the environment:

   ```bash
   conda activate GQ
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

## Project Overview

This project includes a series of Jupyter notebooks and scripts that:

- **Merge and preprocess data** (`1_Data_Merging_final.ipynb`)
- **Perform optimization of modeling parameters** (`2_Optimization_final.ipynb`)
- **Evaluate model performance metrics** (`3_Metrics.ipynb`)
- **Conduct final modeling for geothermal heat flow** (`4_Modelling_final.ipynb`)

The repository also includes utility functions (`GQ_utils.py`) to support the analysis and a requirements file (`requirements.yml`) for easy environment setup.

## License

This project is licensed under the MIT License.
