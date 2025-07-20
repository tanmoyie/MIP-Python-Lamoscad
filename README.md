
# LAMOSCAD Optimization Project ✈️⚙️

This repository contains optimization models and algorithms developed for the **LAMOSCAD** (Location Allocation Modeling to Optimize Spill Coverage and Cost in the Canadian Arctic) project.

The models are designed to solve complex **facility location**, **resource allocation**, and **emergency response optimization** problems using tools such as **Gurobi**, **Python**, and **Geospatial Data**.

---

## 📂 Project Structure

```
MIP-Python-Lamoscad/
│
├── data/                            # Input datasets (e.g., oil spill, station info)
│
├── results/                         # Output results
│   ├── artifacts/                   # Saved output files or model artifacts
│   └── plots/                       # Excel and figure output files (e.g., .xlsx, .png)
│
├── scripts/                         # Experiment scripts for running models and analyses
│   ├── generate_data.py             # Preprocessing script
│   ├── obtain_pcp_data_s4.3.2.py
│   ├── perform_sensitivity_analysis_s4.3.py
│   ├── run_computational_findings_s4.2.py
│   ├── run_lamoscad_mclp_s4.2.3.py
│   └── run_milp_BnC_s4.2.6.py
│
├── src/                             # Core source code
│   ├── config/                      # Configuration and YAML loaders
│   ├── models/                      # Mathematical model definitions
│   ├── preprocessing/               # Data loading and preprocessing utils
│   ├── solvers/                     # Custom solver logic (e.g., branch and cut)
│   ├── utils/                       # General utilities
│   └── visualization/               # Drawing maps, routes, and networks
│
├── requirement.txt
```

---

## 📊 Optimization Techniques Used

- **Mixed Integer Linear Programming (MILP)**
- **Branch-and-Cut Algorithms**
- **Large-Scale Optimization with Gurobi**
- **Sensitivity Analysis**
- **Geospatial Optimization**

---

## 🚀 Key Features

- Gurobi-powered mathematical programming
- Modular & reusable codebase (easy to extend)
- Configurable parameters via YAML files
- Automated result saving & reporting
- Ready-to-run Jupyter notebooks & scripts

---

## ⚙️ Technologies

- Python 🐍
- Gurobi Optimizer
- Pandas, GeoPandas, Shapely, Matplotlib
- YAML Configurations
- Jupyter Notebooks

---

## 📥 Installation

1. Clone this repository:
```bash
git clone https://github.com/your_username/lamoscad-optimization-project.git
cd lamoscad-optimization-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Gurobi license (if required).

---

## 📄 How to Run

- **Scripts**: Execute optimization models from `scripts/`:
```bash
python scripts/run_large_scale.py
```

- **Configurations**: Adjust model parameters inside `src/config/model_config.yaml`.

---

## 📚 References

This project is based on research in operations research and optimization for maritime emergency response systems.

---

## 📢 Acknowledgements

- Gurobi Academic License  
- Python Open Source Libraries  
- MARS team at Dalhousie University 

---

## 📧 Contact

For questions or collaborations, please contact:
**[Tanmoy Das]** — *Operations Research Scientist*  
GitHub: [tanmoiyie](https://github.com/tanmoyie)

---

> **Note:** This repo is intended for academic and research purposes only.
