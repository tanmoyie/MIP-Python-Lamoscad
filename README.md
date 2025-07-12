
# LAMOSCAD Optimization Project ✈️⚙️

This repository contains optimization models and algorithms developed for the **LAMOSCAD** (Large-scale Arctic Maritime Oil Spill Cleanup and Allocation Decision) project.

The models are designed to solve complex **vehicle routing**, **resource allocation**, and **emergency response optimization** problems using tools such as **Gurobi**, **Python**, and **Geospatial Data**.

---

## 📂 Project Structure

```
lamoscad-optimization-project/
│
├── src/                       # Source code (models, solvers, preprocessing)
│├── config/                   # Config loader & YAML config files
│├── preprocessing/            # Data loading & preprocessing utilities
│├── models/                   # Optimization models (Gurobi, MILP, etc.)
│├── solvers/                  # Solver wrappers and algorithms
│├── visualization/            # Map plotting & result visualization
│└── utils/                    # Helper utilities
│
├── notebooks/                 # Jupyter notebooks for experiments
├── data/                      # Input data (ignored by Git)
├── results/                   # Model results (ignored by Git)
├── tests/                     # Unit tests (optional)
├── scripts/                   # CLI-friendly experiment scripts
│
├── README.md                  # This file
├── .gitignore                 # Ignore cache, logs, and data
├── requirements.txt           # Python dependencies
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

- Large-scale optimization models for oil spill response
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

- **Notebooks**: Explore and run Jupyter notebooks inside `notebooks/`.
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
- Texas State University Research Team

---

## 📧 Contact

For questions or collaborations, please contact:
**[Your Name]** — *Operations Research Postdoctoral Researcher*  
Email: your_email@example.com  
GitHub: [your_username](https://github.com/your_username)

---

> **Note:** This repo is intended for academic and research purposes only.
