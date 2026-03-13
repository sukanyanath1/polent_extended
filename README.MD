# Political Entity Analysis Setup

## For running locally

### First Time Steps
1. Clone the repository locally with
    ``` bash
    git clone https://github.com/sukanyanath1/polent_extended.git
    ```
2. Create a virtual environment with python or conda. For example for python, use `python -m venv .venv`
3. Activate the virtual environment with `source .venv/Scripts/activate`
4. Install pip requirements with `pip install -r requirements.txt`
5. Import the data to the project repository.
6. Create a [Huggingface](https://huggingface.co/docs/hub/security-tokens) `HUGGINGFACE_HUB_TOKEN` under your Huggingface account store this in an `.env` file by following the example shown in `.env.example`.
6. Save the data in a Huggingface friendly format under `data/hf_format' by following [Explore Data](#explore-data) section.

### Training 

Example run for park dataset

    ```bash
    MODEL_NAME="bert-base-uncased"
    MODEL_SAVING_PATH="models"
    DATASET_PATH="/data/hf_format/park"
    RESULTS_PATH="results"
    python train.py --model_name "${MODEL_NAME}" --model_saving_path "${MODEL_SAVING_PATH}" --results_path "${RESULTS_PATH}" --dataset_path "${DATASET_PATH}"

    ```

## Running on Ubelix

### First Time Steps
1. Login to Ubelix by following the steps [here](https://hpc-unibe-ch.github.io/firststeps/loggingin/). Note that you must have an account in [Ubelix](https://hpc-unibe-ch.github.io/firststeps/) and request GPU access.
2. Clone the repository in your home directory with
    ``` bash
    git clone https://github.com/sukanyanath1/polent_extended.git
    ```Clone the repository on 
3. Load Anaconda module with `module load Anaconda3`
4. Set up the Conda environment on Ubelix with `conda create --name myenv`.
5. Activate the Conda environment with `conda activate myenv`
6. Install pip requirements from the cloned repo with `pip install -r requirements.txt`
7. Save the data in a Huggingface friendly format under `data/hf_format' by following [Explore Data](#explore-data) section.
8. Send data in the `hf_format` from local workstation to Ubelix with `scp`. For example
    ```bash
    scp -r ~/Polent_Extended/data <YOUR_USERNAME>@submit02.unibe.ch:/storage/homefs/<YOUR_USERNAME>/polent_extended/
    ```
9. Create a `slurm.sh` file inside the repository and populate as shown in `slurm.sh.example`.

### Training 

1. Submit a job on Ubelix. For more information see [here](https://hpc-unibe-ch.github.io/runjobs/scheduled-jobs/submission/). 
    ```bash
    sbatch slurm.sh
    ```
2. Monitor your jobs using the following command. For more information, see [here](https://hpc-unibe-ch.github.io/runjobs/scheduled-jobs/monitoring-jobs/)
    ```bash
    squeue --me
    ```
3. In case of error, view `error.txt` or ` polent_logs.txt` inside the repository.

## Explore data

The notebook `explore_data` prepares the data in HF format for training and saves the data inside the `data/hf_format` folder.

## Results and Intepretation
The results of the training are saved under the `results` folder. If the training was done on Ubelix, copy the results to the local workstation using `scp` command. For example

```bash
scp -r <YOUR_USERNAME>@submit02.unibe.ch:/storage/homefs/<YOUR_USERNAME>/polent_extended/results ~/Polent_Extended/ 
```
Once the results are imported, run the following script locally.

```bash
python visualize_results.py --results_root results --output_dir results_visualizations
```

This produces the following files inside `results_visualizations`

- `summary.csv`  
  Tabular summary of each run (model + dataset), including:
  - validation accuracy
  - test accuracy
  - recomputed array accuracy from predicted vs true labels
  - number of examples
  - source `results.json` path

- `summary.json`  
  JSON version of the same summary information in `summary.csv`.

- `accuracy_overview.png`  
  Bar chart comparing validation and test accuracy for every `(model, dataset)` result found.

- `confusion_matrices/`  
  One confusion matrix image per `(model, dataset)` pair.
  File name pattern:
  - `<model>__<dataset>.png`


