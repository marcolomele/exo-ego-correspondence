# Step-by-Step Guide: Running O-MaMa Training on Bocconi HPC

This guide will walk you through running O-MaMa model training on the university's HPC system.

## Prerequisites
- VPN connection to Bocconi (if required): https://bocconi.sharepoint.com/sites/BocconiStudentsHPC/SitePages/SSH-Login.aspx
- SSH access credentials (username: 3152128)

---

## Step 1: Connect to the HPC Login Node

From your local terminal, SSH into the HPC:

```bash
ssh 3152128@slnode-da.sm.unibocconi.it
```

Enter your password when prompted.

---

## Step 2: Set Up Your Working Directory

Once connected, create a directory structure for your project:

```bash
# Create main project directory
mkdir -p ~/vm-cv/{models,data,logs}

# Navigate to the project directory
cd ~/vm-cv
```

---

## Step 3: Set Up Python Environment

Create and activate a conda environment with the required packages:

```bash
# Load conda module
module load modules/miniconda3

# Create a new environment (or use existing General_Env if preferred)
conda create --name omama_env python=3.9 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate omama_env

# Install required packages from requirements.txt
# First, upload requirements.txt (see Step 4), then:
pip install -r ~/vm-cv/models/O-MaMa/requirements.txt

# OR install packages manually:
# pip install torch==2.4.0 torchvision==0.19.0 matplotlib tqdm opencv-python pycocotools scikit-learn scikit-image segment-anything
```

**Note:** Keep the terminal session open, or remember to activate the environment each time you log in:
```bash
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate omama_env
```

---

## Step 4: Upload Your Files to HPC

**From your LOCAL machine** (open a NEW terminal window, stay in your local project directory):

### 4a. Upload O-MaMa model code
```bash
# Navigate to your local project directory
cd /Users/marcolomele/Documents/Repos/vm-cv

# Upload the entire O-MaMa model directory
scp -r models/O-MaMa/ 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/models/
```

### 4b. Upload data files
```bash
# Upload data directory (this may take a while depending on data size)
# From the project root directory:
cd /Users/marcolomele/Documents/Repos/vm-cv

# Upload the data directory (choose one or both):
# For downscaled data:
scp -r data/health_downscaled_data_omama/ 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/data/

# For normal data:
# scp -r data/health_normal_data_omama/ 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/data/
```

**Note:** The `scp` command syntax is: `scp <local_file> <username>@<host>:<remote_path>`

**Important:** Data uploads can be very large. Consider using `rsync` for resumable transfers:
```bash
rsync -avz --progress data/health_downscaled_data_omama/ 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/data/health_downscaled_data_omama/
```

---

## Step 5: Verify Files Are Uploaded

**Back on the HPC** (in your SSH session), verify everything is in place:

```bash
cd ~/vm-cv

# Check model code
ls -la models/O-MaMa/
ls -la models/O-MaMa/main.py

# Check data (verify the paths match your --root argument)
ls -la data/health_downscaled_data_omama/
ls -la data/health_downscaled_data_omama/dataset_jsons/
ls -la data/health_downscaled_data_omama/Masks_TRAIN_EXO2EGO/
ls -la data/health_downscaled_data_omama/Masks_VAL_EXO2EGO/
```

**Important:** Verify that the data structure matches what the `Masks_Dataset` class expects. The `--root` argument should point to the dataset directory (e.g., `~/vm-cv/data/health_downscaled_data_omama`).

---

## Step 6: Create a SLURM Script

Create a SLURM batch script to run your training:

```bash
nano ~/vm-cv/models/O-MaMa/run_training.sh
```

Paste the following content (adjust resources as needed):

```bash
#!/bin/bash
#SBATCH --job-name=omama_training
#SBATCH --account=3176145
#SBATCH --partition=dsba
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=../../logs/%x_%j.out
#SBATCH --error=../../logs/%x_%j.err

# Load conda module
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate omama_env

# Navigate to O-MaMa directory
cd ~/vm-cv/models/O-MaMa

# Set dataset root path (adjust based on which dataset you're using)
DATA_ROOT=~/vm-cv/data/health_downscaled_data_omama

# Run training with desired arguments
python main.py \
    --root ${DATA_ROOT} \
    --patch_size 14 \
    --context_size 20 \
    --devices 0 \
    --N_masks_per_batch 32 \
    --batch_size 12 \
    --N_epochs 10 \
    --order 2 \
    --exp_name Train_OMAMA_EgoExo_HPC \
    --output_dir ~/vm-cv/models/O-MaMa/train_output

# Optional: Add --reverse flag for exo->ego pairs
# python main.py --root ${DATA_ROOT} --reverse --patch_size 14 ...

echo "Training completed!"

# Deactivate environment
conda deactivate
module unload modules/miniconda3
```

Save and exit:
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter` to save

Make the script executable:
```bash
chmod +x ~/vm-cv/models/O-MaMa/run_training.sh
```

**Note:** Adjust the following parameters as needed:
- `--N_epochs`: Number of training epochs (default: 3, increase for longer training)
- `--batch_size`: Batch size (default: 12, adjust based on GPU memory)
- `--N_masks_per_batch`: Number of masks per batch (default: 32)
- `--mem`: Memory request (64G default, increase if needed)
- `--time`: Time limit (48:00:00 default, increase for longer training)
- `--gres=gpu:1`: Request 1 GPU (adjust if you need more)

---

## Step 7: Submit the Job to SLURM

Submit your job to the SLURM scheduler:

```bash
cd ~/vm-cv/models/O-MaMa
sbatch run_training.sh
```

You should see output like:
```
Submitted batch job 12345
```

**Note the job ID** - you'll use it to check the status.

---

## Step 8: Monitor Your Job

### Check job status:
```bash
squeue -u 3152128
```

### Check job details:
```bash
squeue -j <JOB_ID>
```

### View output in real-time (if job is running):
```bash
tail -f ~/vm-cv/logs/omama_training_<JOB_ID>.out
```

### View errors:
```bash
tail -f ~/vm-cv/logs/omama_training_<JOB_ID>.err
```

### Check GPU usage (if job is running):
```bash
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
```

---

## Step 9: Check Results

Once the job completes, check the results:

```bash
cd ~/vm-cv/models/O-MaMa

# Check training output directory
ls -la train_output/

# Check for model weights
ls -la train_output/run_*/model_weights/

# Check for validation results
ls -la train_output/run_*/val_results_epoch*.json

# Check training statistics
ls -la train_output/run_*/training_stats_*.json

# View training log
cat train_output/run_*/training_*.log
```

---

## Step 10: Download Results (Optional)

**From your LOCAL machine**, download the results:

```bash
# Download entire training output directory
scp -r 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/models/O-MaMa/train_output/ /Users/marcolomele/Documents/Repos/vm-cv/models/O-MaMa/

# Or download specific run directory:
# scp -r 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/models/O-MaMa/train_output/run_* /Users/marcolomele/Documents/Repos/vm-cv/models/O-MaMa/train_output/

# Download logs
scp 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/logs/*.out /Users/marcolomele/Documents/Repos/vm-cv/logs/
scp 3152128@slnode-da.sm.unibocconi.it:~/vm-cv/logs/*.err /Users/marcolomele/Documents/Repos/vm-cv/logs/
```

---

## Troubleshooting

### Job is pending/not starting:
- Check available resources: `sinfo`
- Check GPU availability: `sinfo -o "%.10P %.5a %.10l %.6D %.6t %.8z %.6m %.8d %.6w %.8f %.20G"`
- Check your account limits: `sacct -u 3152128`
- Try reducing requested resources (CPU, memory, time) or removing GPU request if not available

### Job fails immediately:
- Check the error log: `cat ~/vm-cv/logs/omama_training_<JOB_ID>.err`
- Verify Python environment is activated correctly
- Check that all files are uploaded and paths are correct
- Verify the `--root` path points to the correct dataset directory

### Import errors:
- Make sure all Python dependencies are installed: `pip list`
- Verify you're in the correct directory when running the script
- Check that the O-MaMa directory structure is intact

### CUDA/GPU errors:
- Verify GPU is available: `nvidia-smi` (on compute node, not login node)
- Check that PyTorch was installed with CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Try using `--devices cpu` to run on CPU if GPU is not available

### Path errors:
- Verify data paths: The `--root` argument should point to the dataset root (e.g., `~/vm-cv/data/health_downscaled_data_omama`)
- Check that dataset_jsons directory exists and contains the required JSON files
- Verify Masks_TRAIN_EXO2EGO and Masks_VAL_EXO2EGO directories exist

### Out of memory errors:
- Reduce `--batch_size` (e.g., from 12 to 8 or 4)
- Reduce `--N_masks_per_batch` (e.g., from 32 to 16)
- Increase `--mem` in SLURM script

### Need to cancel a job:
```bash
scancel <JOB_ID>
```

---

## Quick Reference Commands

```bash
# Connect to HPC
ssh 3152128@slnode-da.sm.unibocconi.it

# Activate environment
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate omama_env

# Submit job
cd ~/vm-cv/models/O-MaMa
sbatch run_training.sh

# Check status
squeue -u 3152128

# View output
tail -f ~/vm-cv/logs/omama_training_<JOB_ID>.out

# Exit HPC
exit
```

---

## Example Training Commands

### Basic training (ego->exo):
```bash
python main.py \
    --root ~/vm-cv/data/health_downscaled_data_omama \
    --N_epochs 10 \
    --batch_size 12 \
    --exp_name Train_OMAMA_EgoExo
```

### Training with reverse direction (exo->ego):
```bash
python main.py \
    --root ~/vm-cv/data/health_downscaled_data_omama \
    --reverse \
    --N_epochs 10 \
    --batch_size 12 \
    --exp_name Train_OMAMA_ExoEgo
```

### Training with custom output directory:
```bash
python main.py \
    --root ~/vm-cv/data/health_downscaled_data_omama \
    --output_dir ~/vm-cv/models/O-MaMa/custom_output \
    --N_epochs 10
```

---

## Notes

- **Login node limitations**: You can only save files on the login node, not on compute nodes
- **Long-running jobs**: Use SLURM for anything that takes more than a few minutes
- **Resource requests**: Adjust `--cpus-per-task`, `--mem`, `--time`, and `--gres=gpu:1` based on your needs
- **Account number**: Verify your account number (3176145) matches what's in the SLURM script
- **Partition**: The script uses `dsba` partition - verify this is correct for your account
- **GPU availability**: Check if GPUs are available on your partition before requesting them
- **Data size**: The dataset can be very large - consider using `rsync` for resumable uploads
- **Model checkpoints**: The script automatically saves best model and last epoch checkpoints in `train_output/run_*/model_weights/`

