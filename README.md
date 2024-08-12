# Robot Learning from Diverse Human Feedback
 
### Create Conda environment
`conda create --name rlhf`
`conda activate rlhf`

### Install Requirements
`pip install -r requirements.txt`

### Install D4RL
`cd Uni-RLHF-Platform/d4rl
pip install -e .`

### Install hdf5
`conda install anaconda::hdf5`

### Install PyTorch
With GPU
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
CPU only
`pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu`

### Install Mujoco
- Download the MuJoCo library:
`wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz`
- Create the MuJoCo folder:
`mkdir ~/.mujoco`
- Extract the library to the MuJoCo folder:
`tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/`
- Add environment variables (run `nano ~/.bashrc`):
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl`
- Install dependencies:
`conda install -c conda-forge patchelf fasteners cython==0.29.37 cffi pyglfw libllvm11 imageio glew glfw mesalib`