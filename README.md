# Imageomics-Species-ReID
Service supporting reidentification with machine learning for various animal species based on Wildbook's Image Analysis (WBIA) by WildMe
## Requirements
- Python 3.11
- Python dependencies listed in [environment.yaml](environment.yaml)
## Setup on Windows
- Install a Conda-based Python 3 distribution
- Install Windows Subsystem for Linux with `wsl --install` or visit [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) for further instructions
- Activate WSL with `wsl` and navigate to the desired directory
- Install Mambaforge:
```
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -o Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
```
- When you are asked the following question, answer with **yes**:
```
Do you wish the installer to prepend the install location to PATH ...? [yes|no]
```
- Clone the repository:
```
git clone https://github.com/Julian-Bain/Imageomics-Species-ReID.git
cd Imageomics-Species-ReID
```
- Install snakemake and other dependencies within an isolated environment (smk_pipeline can be replaced with an alternative name):
```
conda activate base
mamba env create --name smk_pipeline --file environment.yaml
conda activate smk_pipeline
```
## Alternative Setup
- Clone the repository:
```
git clone https://github.com/Julian-Bain/Imageomics-Species-ReID.git
```
- Full and alternative instructions for installing snakemake can be found in the snakemake documentation ([Installation](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) | [Setup](https://snakemake.readthedocs.io/en/stable/tutorial/setup.html))
## Execution
- Activate WSL with `wsl` and base conda environment with `conda activate base` if not active
- Activate snakemake environment (smk_pipeline can be replaced with an alternative name):
```
conda activate smk_pipeline
```
- Run the pipeline:
```
snakemake -s snakefile.smk --cores 1
```
## GGR-Specific Functions
- Additional script `extrapolate_ggr_gps.py` extrapolates from existing GPS data to fill in for missing GPS data for images from other cameras.
- Run in same environment by providing input and output image data .json paths:
```
python extrapolate_ggr_gps.py /mnt/c/Users/Julian/Research/Pipeline/data/test_dataset/image_data.json /mnt/c/Users/Julian/Research/Pipeline/data/test_dataset/image_data_complete.json
```
## Future Tasks
- Remove dependence of detector on ground truth .csv file.
- Implement threading for image parameter computation and validity checking if image import is too slow.
- Set up pipeline within IDEA cluster and perform larger tests.
