# asymmetric-catalysis-qspr-ml

Machine learning models for predicting enantioselectivity in asymmetric catalysis using small, curated datasets with case studies on magnesium-catalyzed epoxidation and thia-Michael addition.

## Usage

The computational experiments are divided into folders named:
- `classifier` for magnesium-catalyzed epoxidation
- `regression` for magnesium-catalyzed epoxidation  
- `regression_II` for magnesium-catalyzed epoxidation (modified initial input)
- `thia-Michael_regression` for thia-Michael addition with the following subfolders:
  - `first_split`
  - `second_split`
  - `third_split`

The workflow includes examining:
- both the substrate and product of the reaction
- various types of molecular descriptors (such as **Morgan fingerprints**, **mordred descriptors**, **chemBERTa embeddings**, **Circus descriptors**, **Chyline descriptors**, **Sterimol descriptors**, and **descriptors derived from Gasteiger charges**)

Each of the main folders include:
- **Data** subfolder, with starting, validation and experimental data
- **module** subfolder where three project's important scripts are located, namely `calculate_descriptors.py`, `hyperparameters.py` and `models_creation.py`
- **QSPR** subfolder where the main execution file is stored, named `QSPR_analysis.py` with `config.yaml`, which drives the different modes of the computational experiment
- **Results** folder contains subfolders `logs` with logging from the calculations, `Selected_Data` with train and test sets for each experiment, and `Trained_Models` with all optimized models saved.

## The used libraries are (requirements, 12.04.2026):

conda create --name cheminf_gpu python=3.11 -y
conda activate cheminf_gpu
conda install -c conda-forge pandas=2.3.3 scikit-learn=1.8.0 matplotlib joblib threadpoolctl scipy pyyaml pyarrow fastparquet=2025.12.0 -y
pip install rdkit==2025.9.3 mordred==1.2.0 mordredcommunity==2.0.6 chython==2.13 doptools==1.3.9 morfeus-ml==0.8.0 transformers torch sentencepiece==0.2.1 xgboost==3.1.2 hyperopt==0.2.7 xlsxwriter==3.2.9 openpyxl==3.1.5 notebook jupyterlab ipykernel
python -m ipykernel install --user --name cheminf_gpu --display-name "cheminf_gpu (QSPR)"