# Stranger Danger! Identifying and Avoiding Unpredictable Pedestrians in RL-based Social Robot Navigation

## 0) Setup
1. Create an environment with Python 3.6 on Ubuntu Linux.
2. Install the [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.

	Note: Make sure that CMake and the tested version of Cython are installed. If the build fails, you may need to delete the build folder and try building again.

3. Within the main `stranger-danger` directory, run the following command:
```
pip install -e .
```

## 1) Train Uncertainty Estimation MLPs

### 1.1) Collect Pedestrian Walking Data

From the `uncertainty` folder, run the following command to collect data of randomized ORCA pedestrians navigating through 100 episodes of six different scenarios:
```
python collect_data.py --vary_param epsilon --num_episodes 100 --output_dir data/
```

This will create CSV files of relevant data in a folder called `data` within the `uncertainty` folder.

### 1.2) Preprocess Pedestrian Walking Data

From the `uncertainty` folder, run the following command to preprocess the collected pedestrian walking data to prepare it to be used as input to the uncertainty network:
```
python preprocess_data.py --data_dir data/
```

This will generate several folders of CSV files within a folder called `preprocessed_data` in the `data` folder you created in the previous step.

### 1.3) Train Uncertainty Networks

From the `uncertainty` folder, run the following script to train and save 20 uncertainty prediction networks with 20 different numbers of time steps:
```
chmod +x train_models.sh 
./train_models.sh
```

This will train 20 models using the preprocessed data from the previous step and store each model within its own folder in a larger folder called `models`. The model folders are named such that `uncertain_T` is the model trained using T time steps. Within this folder, there is also a plot of the loss curve and a plot of the model prediction accuracy. *Note: It may take several hours or days to train all 20 models.*

## 2) Train an RL Policy with Known Uncertainty

### 2.1) Train an RL Policy

From the `control` folder, run the following commands to train a socially-aware RL policy with ground-truth uncertainty values using Noisy ORCA pedestrians:
```
python train.py --policy uncertain_sarl --output_dir model/
```

This will save a trained uncertainty-aware RL policy in a folder called `model` in the `control` folder. The model will be trained using the parameters specified in the configuration files `env.config`, `policy.config`, and `train.config` within the `configs` folder. More information on the parameters in these configuration files can be found [here](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md).

### 2.2) Plot Policy Performance

To plot the success rate, collision rate, and timeout rate of your recently trained model during training, run the following command from within the `control` folder:
```
python utils/plot.py model/ --plot_sr --plot_cr --plot_tr
```

This will save three plots within your RL policy `model` directory.

## 3) Evaluate the Trained RL Policy

### 3.1) Evaluate Policy with Noisy ORCA

The following command with run 500 random trials evaluating the performance of your trained RL policy on Noisy ORCA pedestrians with a maximum unpredictability score of 0.5:
```
python test.py --policy uncertain_sarl --model_dir model/ --estimate_eps --max_epsilon 0.5 --stats_file model/stats.csv
```

This will save a CSV file called `stats.csv` within the `model` folder storing your RL policy. Evaluation parameters can be set in the configuration file `model/env.config`. More information on the parameters in these configuration files can be found [here](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md).

### 3.2) Visualize Trials of Policy Evaluations

The following command will allow you to visualize a single evaluation trial:
```
python test.py --policy uncertain_sarl --model_dir model/ --estimate_eps --max_epsilon 0.5 --visualize --test_case 1 --video_file videos/test1.mp4
```

This will save a video called `test1.mp4` in the `videos` folder showing trial number 1.
