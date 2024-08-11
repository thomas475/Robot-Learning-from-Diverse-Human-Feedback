# Robot Learning from Diverse Human Feedback
 
### Create Conda environment
`conda create --name rlhf`
`conda activate rlhf`

### Install hdf5
`conda install anaconda::hdf5`

### Install PyTorch
With GPU
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
CPU only
`pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu`

### Install Mujoco
`conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3`
in bashrc add
'export CPATH=$CONDA_PREFIX/include'