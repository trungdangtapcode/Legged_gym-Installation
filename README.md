# Legged_gym Installation Guide


This guide outlines the installation process for the [legged_gym](https://github.com/leggedrobotics/legged_gym) and [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) repositories, incorporating lessons learned from Kaggle setup experiences. It assumes you have Anaconda and an NVIDIA GPU driver installed. Always verify your GPU driver compatibility using `nvidia-smi`.

## 1. Set Up the Conda Environment

1. **Create a Conda environment with Python 3.8**:
   ```bash
   conda create -n legged_gym python=3.8
   conda activate legged_gym
   ```

2. **Install required system libraries**:
   ```bash
   sudo apt-get update
   sudo apt install libpython3.8
   ```

3. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
   *Note*: If you encounter CUDA-related issues, ensure your GPU driver and CUDA versions are compatible. Check with `nvidia-smi`.

## 2. Install Isaac Gym

1. **Download Isaac Gym Preview 4**:
   - Visit [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym), sign in, and agree to the terms.
   - Obtain the direct download URL for Isaac Gym Preview 4 and download it:
     ```bash
     cd ~
     wget "PASTE_DIRECT_DOWNLOAD_URL_HERE" -O IsaacGym_Preview_4_Package.tar.gz
     tar -xf IsaacGym_Preview_4_Package.tar.gz
     ```

2. **Set up the Isaac Gym environment**:
   - Navigate to the Isaac Gym directory and create the provided `rlgpu` environment:
     ```bash
     cd ~/isaacgym
     ./create_conda_env_rlgpu.sh
     ```
   - Activate the `rlgpu` environment (optional, I still remain `legged_gym` environment):
     ```bash
     conda activate rlgpu
     ```

3. **Install Isaac Gym**:
   ```bash
   cd ~/isaacgym/python
   pip install -e .
   ```

4. **Test Isaac Gym**:
   ```bash
   cd ~/isaacgym/examples
   python 1080_balls_of_solitude.py
   ```
   - If you encounter `ImportError: libpython3.8.so.1.0: cannot open shared object file`, reinstall the library:
     ```bash
     sudo apt install libpython3.8
     ```

5. **Install gym library** (required for Isaac Gym compatibility):
   ```bash
   pip install gym
   ```

## 3. Install IsaacGymEnvs

1. **Clone and install IsaacGymEnvs**:
   ```bash
   cd ~
   git clone https://github.com/isaac-sim/IsaacGymEnvs.git
   cd IsaacGymEnvs
   pip install -e .
   ```

2. **Fix NumPy deprecation errors**:
   - If you encounter `AttributeError: module 'numpy' has no attribute 'float'` or similar issues due to deprecated NumPy aliases, install a compatible NumPy version:
     ```bash
     pip install numpy==1.23.0
     ```
   - Alternatively, add the following to your code or script to handle deprecated types:
     ```python
     import numpy as np
     np.float = np.float32
     np.int = np.int32
     ```
   - *Reference*: [NumPy 1.20.0 release notes](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations).

## 4. Install rsl_rl (PPO Implementation)

1. **Clone and install rsl_rl (version 1.0.2)**:
   ```bash
   cd ~
   git clone https://github.com/Lab-of-AI-and-Robotics/Legged_gym-Installation
   cd rsl_rl
   git checkout v1.0.2
   pip install -e .
   ```

2. **Handle missing Git**:
   - If Git is not installed, run:
     ```bash
     sudo apt install git
     ```

## 5. Install legged_gym

1. **Clone and install legged_gym**:
   ```bash
   cd ~
   git clone https://github.com/leggedrobotics/legged_gym
   cd legged_gym
   pip install -e .
   ```

2. **Test the installation**:
   ```bash
   python legged_gym/scripts/train.py --task=anymal_c_flat
   ```

3. **Resolve common errors**:
   - **ModuleNotFoundError: No module named 'tensorboard'**:
     ```bash
     pip install tensorboard
     ```
   - **AttributeError: module 'distutils' has no attribute 'version'**:
     ```bash
     pip install setuptools==59.5.0
     ```

## 6. Install walk-these-ways (Advanced legged_gym)

1. **Clone and install walk-these-ways**:
   ```bash
   cd ~
   git clone https://github.com/Improbable-AI/walk-these-ways
   cd walk-these-ways
   pip install -e .
   ```

## Troubleshooting

- **GPU driver issues**:
  - Run `nvidia-smi` to verify your GPU driver and CUDA version. Update your driver if necessary.
- **Python version conflicts**:
  - Ensure all installations use the `legged_gym` or `rlgpu` Conda environment with Python 3.8.
- **ModuleNotFoundError: No module named 'isaacgym'**:
  - Reinstall `gym` in both the home and Isaac Gym directories:
    ```bash
    pip install gym
    cd ~/isaacgym/python
    pip install gym
    ```

This guide incorporates Kaggle-based setup experiences, streamlining the process while addressing common errors. If you encounter issues, double-check your environment and dependencies.
