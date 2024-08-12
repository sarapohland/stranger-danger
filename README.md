# Stranger Danger

This is the codebase for the paper titled ["Stranger Danger! Identifying and Avoiding Unpredictable Pedestrians in RL-based Social Robot Navigation,"](https://ieeexplore.ieee.org/document/10610413) which was presented at the 2024 IEEE International Conference on Robotics and Automation (ICRA). This README describes how to reproduce the results achieved in this paper. An extended version of our paper is available on [arXiv](https://arxiv.org/abs/2407.06056), and a video showcasing our methods and results is available on [YouTube](https://youtu.be/9IDhXvCC58w?si=Y0Di3d5NjWj-3nvl). If you find this work useful, please cite our paper using the citation provided at end of this [README](https://github.com/sarapohland/stranger-danger#citing-our-work).

## 0) Setup

1. Create an environment with Python 3.6 on Ubuntu Linux.
2. Install the [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.

	Note: Make sure that CMake and the tested version of Cython are installed. If the build fails, you may need to delete the build folder and try building again.

3. Within the main `stranger-danger` directory, run the following command:
```
pip install -e .
```

## 1) Train Uncertainty Estimation Networks

The following section details how to train the set of uncertainty estimation networks utilized by two variations of the uncertainty-aware RL policy.

### 1.1) Collect pedestrian walking data

From the `uncertainty` folder, run the following command to collect data of randomized ORCA pedestrians navigating through 100 episodes of six different scenarios:
```
python collect_data.py --vary_param epsilon --num_episodes 100 --output_dir data/
```

This will create CSV files of relevant data in a folder called `data` within the `uncertainty` folder.

### 1.2) Preprocess pedestrian walking data

From the `uncertainty` folder, run the following command to preprocess the collected pedestrian walking data to prepare it to be used as input to the uncertainty network:
```
python preprocess_data.py --data_dir data/
```

This will generate several folders of CSV files within a folder called `preprocessed_data` in the `data` folder you created in the previous step.

### 1.3) Train networks with preprocessed data

From the `uncertainty` folder, run the following script to train and save 20 uncertainty prediction networks with 20 different numbers of time steps:
```
chmod +x train_models.sh 
./train_models.sh
```

This will train 20 models using the preprocessed data from the previous step and store each model within its own folder in a larger folder called `models`. The model folders are named such that `uncertain_T` is the model trained using T time steps. Within this folder, there is also a plot of the loss curve and a plot of the model prediction accuracy. *Note: It may take several hours or days to train all 20 models.*

## 2) Train Socially-Aware RL Policies

The following section details how to train the baseline SARL policy and the three variations of this RL policy.

### 2.1) Train baseline SARL policy (*SARL*)

From the `control` folder, run the following command to train the baseline SARL policy without uncertainty-awareness using standard ORCA pedestrians:
```
python train.py --policy sarl --output_dir models/sarl/ --env_config configs/env-train.config --policy_config configs/sarl/policy.config --train_config configs/sarl/train.config
```

This will save the trained RL policy to the folder `models/sarl/` in the `control` directory. The model will be trained using the parameters specified in the configuration files (`env_config`, `policy_config`, and `train_config`). More information on the parameters in these configuration files can be found in the configs [README](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md). You can try training with different discomfort distances by changing the `discomfort_dist` value under `reward` in the [training environment config file](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/env-train.config) *Note: It may take several hours or days to train a single model.*

### 2.2) Train policy with modified training process (*Training*)

From the `control` folder, run the following command to train the RL policy without uncertainty-awareness using Noisy ORCA pedestrians:
```
python train.py --policy sarl --output_dir models/training/ --env_config configs/env-train.config --policy_config configs/sarl/policy.config --train_config configs/sarl/train.config
```

This will save the trained RL policy to the folder `models/training/` in the `control` directory. The model will be trained using the parameters specified in the configuration files (`env_config`, `policy_config`, and `train_config`). More information on the parameters in these configuration files can be found in the configs [README](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md). *Note: It may take several hours or days to train a single model.*

### 2.3) Train policy with modified model architecture (*Model*)

From the `control` folder, run the following command to train the uncertainty-aware RL policy with only the modified model architecture using Noisy ORCA pedestrians:
```
python train.py --policy uncertain_sarl --output_dir models/model/ --env_config configs/env-train.config --policy_config configs/sarl/policy.config --train_config configs/sarl/train.config
```

This will save the trained RL policy to the folder `models/model/` in the `control` directory. The model will be trained using the parameters specified in the configuration files (`env_config`, `policy_config`, and `train_config`). More information on the parameters in these configuration files can be found in the configs [README](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md). *Note: It may take several hours or days to train a single model.*

### 2.4) Train policy with modified reward function (*Reward*)

From the `control` folder, run the following command to train the uncertainty-aware RL policy with the undertainty-dependent reward function using Noisy ORCA pedestrians:
```
python train.py --policy uncertain_sarl --output_dir models/reward/ --env_config configs/env-train.config --policy_config configs/sarl/policy.config --train_config configs/sarl/train.config
```

This will save the trained RL policy to the folder `models/reward/` in the `control` directory. The model will be trained using the parameters specified in the configuration files (`env_config`, `policy_config`, and `train_config`). More information on the parameters in these configuration files can be found in the configs [README](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md). *Note: It may take several hours or days to train a single model.*

## 3) Ablation Study on Noisy ORCA Pedestrians

The following section describes how to reproduce the results provided in the ablation study on noisy ORCA pedestrians.Evaluation parameters can be set in the configuration file `configs/env-noisy.config`. More information on the parameters in this configuration files can be found in the configs [README](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md).

### 3.1) Evaluate baseline SARL policy (*SARL*)

The following command will run 500 random trials evaluating the performance of the baseline SARL policy on Noisy ORCA pedestrians with a maximum unpredictability score of 0.5:
```
python test.py --policy sarl --model_dir models/sarl/ --estimate_eps --max_epsilon 0.5 --stats_file models/sarl/stats-noisy.csv --env_config configs/env-noisy.config
```

This script will save a CSV file called `stats-noisy.csv` within the `models/sarl/` folder.

### 3.2) Evaluate policy with modified training process (*Training*)

The following command will run 500 random trials evaluating the performance of the *Training* policy on Noisy ORCA pedestrians with a maximum unpredictability score of 0.5:
```
python test.py --policy sarl --model_dir models/training/ --estimate_eps --max_epsilon 0.5 --stats_file models/training/stats-noisy.csv --env_config configs/env-noisy.config
```

This script will save a CSV file called `stats-noisy.csv` within the `models/training/` folder.

### 3.3) Evaluate policy with modified model architecture (*Model*)

The following command will run 500 random trials evaluating the performance of the *Model* policy on Noisy ORCA pedestrians with a maximum unpredictability score of 0.5:
```
python test.py --policy uncertain_sarl --model_dir models/model/ --estimate_eps --max_epsilon 0.5 --stats_file models/model/stats-noisy.csv --env_config configs/env-noisy.config
```

This script will save a CSV file called `stats-noisy.csv` within the `models/model/` folder. 

### 3.4) Evaluate policy with modified reward function (*Reward*)

The following command will run 500 random trials evaluating the performance of the *Reward* policy on Noisy ORCA pedestrians with a maximum unpredictability score of 0.5:
```
python test.py --policy uncertain_sarl --model_dir models/reward/ --estimate_eps --max_epsilon 0.5 --stats_file models/reward/stats-noisy.csv --env_config configs/env-noisy.config
```

This script will save a CSV file called `stats-noisy.csv` within the `models/reward/` folder. 

### 3.5) Evaluate baseline ORCA policy (*ORCA*)

The following command will run 500 random trials evaluating the performance of the baseline ORCA policy on Noisy ORCA pedestrians with a maximum unpredictability score of 0.5:
```
python test.py --policy orca --estimate_eps --max_epsilon 0.5 --stats_file models/orca/stats-noisy.csv --env_config configs/env-noisy.config
```

This script will save a CSV file called `stats-noisy.csv` within the `models/orca/` folder. 

### 3.6) Create table comparing policy performance

To compare the performance of these four RL policies (plus the ORCA policy), you can use the compare script in the `control` directory:
```
python compare.py --files models/orca/stats-noisy.csv models/sarl/stats-noisy.csv models/training/stats-noisy.csv models/model/stats-noisy.csv models/reward/stats-noisy.csv --names ORCA SARL Training Model Reward
```

This command will print a table of results for the ablation study with LaTeX formating.

### 4) Abalation Study on Diverse, Realistic Pedestrians

The following section describes how to reproduce the results provided in the ablation study on diverse, realistic pedestrians. Evaluation parameters can be set in the configuration file `configs/env-policies.config`. More information on the parameters in this configuration files can be found in the configs [README](https://github.com/sarapohland/stranger-danger/blob/main/control/configs/README.md).

### 4.1) Evaluate baseline SARL policy (*SARL*)

The following command will run 100 random trials evaluating the performance of the baseline SARL policy on pedestrians operating under various policies (standard ORCA, CADRL, and Linear with randomized parameters):
```
python test.py --policy sarl --model_dir models/sarl/ --estimate_eps --stats_file models/sarl/stats-policies.csv --env_config configs/env-policies.config
```

This script will save a CSV file called `stats-policies.csv` within the `models/sarl/` folder. 

### 4.2) Evaluate policy with modified training process (*Training*)

The following command will run 100 random trials evaluating the performance of the *Training* policy on pedestrians operating under various policies (standard ORCA, CADRL, and Linear with randomized parameters):
```
python test.py --policy sarl --model_dir models/training/ --estimate_eps --stats_file models/training/stats-policies.csv --env_config configs/env-policies.config
```

This script will save a CSV file called `stats-policies.csv` within the `models/training/` folder. 

### 4.3) Evaluate policy with modified model architecture (*Model*)

The following command will run 100 random trials evaluating the performance of the *Model* policy on pedestrians operating under various policies (standard ORCA, CADRL, and Linear with randomized parameters):
```
python test.py --policy uncertain_sarl --model_dir models/model/ --estimate_eps --stats_file models/model/stats-policies.csv --env_config configs/env-policies.config
```

This script will save a CSV file called `stats-policies.csv` within the `models/model/` folder. 

### 4.4) Evaluate policy with modified reward function (*Reward*)

The following command will run 100 random trials evaluating the performance of the *Reward* policy on pedestrians operating under various policies (standard ORCA, CADRL, and Linear with randomized parameters):
```
python test.py --policy uncertain_sarl --model_dir models/reward/ --estimate_eps --stats_file models/reward/stats-policies.csv --env_config configs/env-policies.config
```

This script will save a CSV file called `stats-policies.csv` within the `models/reward/` folder. 

### 4.5) Evaluate baseline ORCA policy (*ORCA*)

The following command will run 100 random trials evaluating the performance of the baseline ORCA policy on pedestrians operating under various policies (standard ORCA, CADRL, and Linear with randomized parameters):
```
python test.py --policy orca --estimate_eps --stats_file models/orca/stats-policies.csv --env_config configs/env-policies.config
```

This script will save a CSV file called `stats-policies.csv` within the `models/orca/` folder. 

### 4.6) Create table comparing policy performance

To compare the performance of these four RL policies (plus the ORCA policy), you can use the compare script in the `control` directory:
```
python compare.py --files models/orca/stats-policies.csv models/sarl/stats-policies.csv models/training/stats-policies.csv models/model/stats-policies.csv models/reward/stats-policies.csv --names ORCA SARL Training Model Reward
```

This command will print a table of results for the ablation study with LaTeX formating.

## 5) Visualize the Trained RL Policies

### 5.1) Visualize trials of baseline SARL policy (*SARL*)

The following command will allow you to visualize a single evaluation trial of the baseline SARL policy:
```
python test.py --policy sarl --model_dir models/sarl/ --estimate_eps --max_epsilon <epsilon_value> --visualize --test_case 1 --video_file videos/sarl/test1.mp4 --env_config <config_file>
```

This will save a video called `test1.mp4` in the `videos/sarl/` folder showing trial number 1 with the environment configurations specified by `config_file` and the maximum epsilon specified by `epsilon_value`.

### 5.2) Visualize trials of policy with modified training process (*Training*)

The following command will allow you to visualize a single evaluation trial of the *Training* policy:
```
python test.py --policy sarl --model_dir models/training/ --estimate_eps --max_epsilon <epsilon_value> --visualize --test_case 1 --video_file videos/training/test1.mp4 --env_config <config_file>
```

This will save a video called `test1.mp4` in the `videos/training/` folder showing trial number 1 with the environment configurations specified by `config_file` and the maximum epsilon specified by `epsilon_value`.

### 5.3) Visualize trials of policy with modified model architecture (*Model*)

The following command will allow you to visualize a single evaluation trial of the *Model* policy:
```
python test.py --policy uncertain_sarl --model_dir models/model/ --estimate_eps --max_epsilon <epsilon_value> --visualize --test_case 1 --video_file videos/model/test1.mp4 --env_config <config_file>
```

This will save a video called `test1.mp4` in the `videos/model/` folder showing trial number 1 with the environment configurations specified by `config_file` and the maximum epsilon specified by `epsilon_value`.

### 5.4) Visualize trials of policy with modified reward function (*Reward*)

The following command will allow you to visualize a single evaluation trial of the *Reward* policy:
```
python test.py --policy uncertain_sarl --model_dir models/reward/ --estimate_eps --max_epsilon <epsilon_value> --visualize --test_case 1 --video_file videos/reward/test1.mp4 --env_config <config_file>
```

This will save a video called `test1.mp4` in the `videos/reward/` folder showing trial number 1 with the environment configurations specified by `config_file` and the maximum epsilon specified by `epsilon_value`.

## Citing Our Work

If you find this codebase useful, please cite the paper associated with this repository:

S. Pohland, A. Tan, P. Dutta and C. Tomlin, "Stranger Danger! Identifying and Avoiding Unpredictable Pedestrians in RL-based Social Robot Navigation," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 15217-15224, doi: 10.1109/ICRA57147.2024.10610413.

@InProceedings{stranger-danger, \
&emsp; author="Pohland, Sara and Tan, Alvin and Dutta, Prabal and Tomlin, Claire", \
&emsp; title="Stranger Danger! Identifying and Avoiding Unpredictable Pedestrians in RL-based Social Robot Navigation", \
&emsp; booktitle="2024 IEEE International Conference on Robotics and Automation (ICRA)", \
&emsp; year="2024", \
&emsp; month="May", \
&emsp; publisher="IEEE", \
&emsp; pages="15217--15224", \
&emsp; doi="10.1109/ICRA57147.2024.10610413" \
&emsp; }