# HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery

This is the official implementation of the IROS paper "[**HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery**](https://arxiv.org/abs/2502.20061)".

This project builds upon the [**Booster Gym**](https://github.com/BoosterRobotics/booster_gym) project, a reinforcement learning (RL) framework designed for humanoid robot locomotion developed by Booster Robotics.

## Installation

Follow these steps to set up your environment:

1. **Install Miniconda**

    Miniconda is a lightweight tool for managing packages and environments.

    ```sh
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    vim ~/.bashrc  # Add the following line
    source ~/miniconda3/bin/activate
    ```

    Create a Python 3.8 environment:

    ```sh
    conda create --name <env_name> python=3.8
    ```

2. **Install PyTorch**

    Activate the environment and install PyTorch with CUDA support:

    ```sh
    conda activate <env_name>
    conda install numpy=1.21.6 pytorch=2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. **Install Isaac Gym**

    Download Isaac Gym from [NVIDIA’s website](https://developer.nvidia.com/isaac-gym/download).

    Extract and install:

    ```sh
    tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    cd isaacgym/python
    pip install -e .
    ```

    Configure the environment for shared libraries:

    ```sh
    cd $CONDA_PREFIX
    mkdir -p ./etc/conda/activate.d
    vim ./etc/conda/activate.d/env_vars.sh  # Add the following lines
    export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
    mkdir -p ./etc/conda/deactivate.d
    vim ./etc/conda/deactivate.d/env_vars.sh  # Add the following lines
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
    unset OLD_LD_LIBRARY_PATH
    ```

4. **Install Additional Requirements**

    Install the required Python dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configurations

Configurations are loaded from `envs/<task>.yaml`. You can override config values using command-line arguments.

**Command-Line Arguments:**

- `--checkpoint`: Path to the model checkpoint.
- `--num_envs`: Number of environments to create.
- `--headless`: Run without a viewer window.
- `--sim_device`: Device for physics simulation (e.g., `cuda:0`, `cpu`).
- `--rl_device`: Device for the RL algorithm (e.g., `cuda:0`, `cpu`).
- `--seed`: Random seed.
- `--max_iterations`: Maximum training iterations.

### Stage 1 Training

To train a basic fall recovery policy, run:

```sh
python train.py --task=T1FallRecovery
```

This trains a policy for the `T1FallRecovery` task using the default configuration. Example configurations are available in `envs/example_cfgs/T1FallRecovery_cfg1.yaml`.

### Stage 2 Training

After completing the basic policy training, proceed to train a more complex policy. Replace `envs/T1FallRecovery.yaml` with `envs/example_cfgs/T1FallRecovery_cfg2.yaml`, and set the `checkpoint` argument to the path of the first-stage trained model.

```sh
python train.py --task=T1FallRecovery
```

To enable network expansion, use the `convert_fall_recovery.py` script. After expanding the model, train the policy with additional control DoFs by specifying the `checkpoint` argument with the converted model path:

```sh
python train.py --task=T1FallRecoveryRandom
```

Example configurations are available in `envs/example_cfgs/T1FallRecoveryRandom_cfg.yaml`. You can add more initial states by modifying the `init_state` list in the configuration file to enhance the robustness of the policy.

### Testing

To test a trained policy in Isaac Gym, run:

```sh
python play.py --task=TASK_NAME --checkpoint=logs/<date-time>/nn/<checkpoint_name>.pth
```

Videos are saved in `videos/<date-time>.mp4` by default. Disable video recording in the config file if needed.

For simulation-to-simulation testing in Mujoco, use:

```sh
python play_mujoco.py --task=T1FallRecovery
```

or

```sh
python play_mujoco_extended.py --task=T1FallRecoveryRandom
```

## Citation
If you find this project useful, please cite our paper:

```bibtex
@article{hifar2025,
      title={HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery},
      author={Chen, Penghui and Wang, Yushi and Luo, Changsheng and Cai, Wenhan and Zhao, Mingguo},
      journal={arXiv preprint arXiv:2502.20061},
      year={2025}
}
```