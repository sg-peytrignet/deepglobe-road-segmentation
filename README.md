
![Road Segmentation](doc/horizontal_road_ribbon.png)

# Automatic road extraction from satellite images

##  Quick Links

> - [ Overview](#overview)
> - [ Features](#features)
> - [ Repository Structure](#repository-structure)
> - [ Modules](#modules)
> - [ Getting Started](#getting-started)
>   - [ Installation](#installation)
>   - [ Running analysis](#running-analysis)
> - [ Contributing](#contributing)
> - [ License](#license)
> - [ Acknowledgments](#acknowledgments)

---

##  Overview

Overview of the analysis, background, data, etc.

---

##  Repository Structure

```sh
deepglobe-road-segmentation
├── LICENSE
├── README.md
├── deep_globe_seg
│   └── helpers.py
├── environment.yml
├── logs
│   └── training_log.csv
├── notebooks
│   └── deep-globe-road-segmentation.ipynb
└── saved_models
    └── unet.weights.h5
```

---

##  Notebooks

###  `deep-globe-road-segmentation.ipynb`

Summary of the analysis and main findings.

---

##  Getting Started

***Requirements***

Ensure you have conda installed on your system before creating the environment. You can refer to the official conda documentation for [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/).

Ensure you have the following dependencies installed on your system:

- **Python**: `3.9`
- **JupyterNotebook**: `version v7.1.3 `

###  Installation

1. Clone the deepglobe-road-segmentation repository:

```sh
git clone git@github.com:sg-peytrignet/deepglobe-road-segmentation.git
```

2. Change to the project directory:

```sh
cd deepglobe-road-segmentation
```

3. Install the dependencies with conda:

```sh
conda env create -f environment.yml
conda activate road_extraction_env
```

###  Running analysis

Run each notebook in the IDE of your choice, such as JupyterLab or Visual Studio. You may launch Jupyter from the terminal:

```sh
jupyter notebook
```

In the Jupyter interface that opens in your web browser, click on the .ipynb file to open it. You can now run cells, modify code, and interact with the notebook.

---
##  License

This project is protected under the [MIT](LICENSE) License.

---

##  Acknowledgments

- packages, kaggle, etc

[**Return**](#quick-links)