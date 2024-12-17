
## Project Overview

This project contains data files in CSV format, each representing the x-y center-of-mass coordinates of a single worm across different experimental conditions. The dataset is organized into two main experimental categories:

1. **Lifespan Experiment**
2. **Optogenetics Experiment**

Each experiment explores different aspects of worm behavior under specific conditions. Below are detailed descriptions of the file structure and the questions that the data seeks to answer.

---

### Data Placement

All data should be placed in the `./data` folder within the project directory. Ensure that the folder structure is preserved as described below for the experiments.

---

### 1. Lifespan Experiment

#### Data Structure
- Located in `./data/Lifespan`.
- Contains four subfolders:
  - `companyDrug`: Data from worms exposed to a specific drug.
  - `control`: Data from worms without drug exposure.
  - `Terbinafin`: Data from worms exposed to Terbinafin.
  - `controlTerbinafin`: Data from worms without drug exposure.
-  The time difference between consecutive frames is 2 seconds. Each recording session lasts for 900 frames (30 minutes), and between each recording session, there is a 5 and a half hour gap. The frame numbering restarts after 10799 to help with organization when working with the equipment.

#### Experimental Context
The `companyDrug` did not appear to have an effect on lifespan, whereas the `Terbinafin` did.

#### Research Question
- Can we predict worm lifespan based on their early behavior?

---

### 2. Optogenetics Experiment

#### Data Structure
- Located in `./data/Optogenetics`.
- Contains two subfolders:
  - `ATR+`: Data from worms with functional optogenetic systems in their neurons.
  - `ATR-`: Data from control worms without functional optogenetic systems.

#### Data Details
- CSV files include an additional column, `Light_Pulse`:
  - Value `1` indicates when the light was turned on.
  - Value `0` indicates when the light was off.

#### Experimental Context
The experiment aims to investigate the effects of light pulses on worm behavior. Specifically, it examines:
- Persistent behavioral differences between worms (worm personalities).
- Variability in responses to light stimulation.

#### Research Question
- Can worm personalities be identified, including differences in response to the light stimulus?

#### Reference
For related research on worm behavior and optogenetics, see the study: [Worm Personalities and Light Response](https://pubmed.ncbi.nlm.nih.gov/29198526/).

---

### Setup Instructions

#### 1. Install Conda
Ensure that you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

#### 2. Create a Conda Environment
Use the provided `environment.yml` file to create a Conda environment named `animal-drug-detection`:

```bash
conda env create -f environment.yml
```

#### 3. Activate the Environment
After the environment is created, activate it using:

```bash
conda activate animal-drug-detection
```

---

### How to Use the Data
1. **Behavior Analysis**: Examine movement patterns across different conditions and timeframes.
2. **Comparative Studies**:
   - Compare worm behavior between `companyDrug` and `control` subfolders in the lifespan experiment.
   - Analyze differences in response to light between `ATR+` and `ATR-` groups in the optogenetics experiment.
3. **Identify Patterns**: Look for behavioral trends, consistent traits, or unique responses to experimental conditions.

---
