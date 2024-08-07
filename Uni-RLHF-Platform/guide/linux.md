# Installation on Linux (Ubuntu)

1. Clone the repo
   ```sh
   git clone https://github.com/TJU-DRL-LAB/Uni-RLHF.git
   cd Uni-RLHF
   ```
2. Setup Anaconda environment
    ```sh
    conda create -n rlhf python==3.9
    conda activate rlhf
    pip install -r requirements.txt
    ```
3. Install NPM packages
   ```sh
   cd uni_rlhf/vue_part
   npm install
   ```
4. Install Redis
   ```sh
   sudo apt-get install redis
   ```
5. Configure the MySQL Database (adjust [user] and [password] accordingly)
   ```
   mysql -u [user] -p
   ```
   Then, in the MySQL environment enter:
   ```
   CREATE DATABASE uni_rlhf;
   exit
   ```
   Afterwards, navigate to ``scripts/create_table.py`` and update the ``cfg`` variable:
   ```
   cfg = {
       'host': 'localhost',
       'port': 3306,
       'username': [user],
       'password': [password],
       'database_name': 'uni_rlhf'
   }
   ```
   Then, execute the script:
   ```
   cd scripts
   py create_table.py
   ```
   Finally, go to `uni_rlhf/config.py` and set ``app.config['SQLALCHEMY_DATABASE_URI']`` to ``'mysql://[user]:[password]@localhost/uni_rlhf'``.
   
### MuJoCo

Many of the datasets use MuJoCo as environment, so it should be installed, too. See [this](https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da) for further details.
* Download the MuJoCo library from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz).
* Create the following folder (adjust [user] accordingly):
    ```sh
    mkdir /home/[user]/.mujoco
    ```
* Extract the library to the .mujoco folder.
    ```sh
    cd ~/Downloads
    tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
    ```
* Update the .bashrc file.
    ```sh
    nano ~/.bashrc
    ```
    Add this to the file (adjust [user] accordingly).
    ```
    export LD_LIBRARY_PATH=/home/[user]/.mujoco/mujoco210/bin 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
    export PATH="$LD_LIBRARY_PATH:$PATH" 
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    ```
* Reload the .bashrc file to register the changes.
    ```sh
    source ~/.bashrc
    ```
* Test that the library is installed.
    ```
    cd ~/.mujoco/mujoco210/bin
    ./simulate ../model/humanoid.xml
    ``` 

### Datasets

Uni-RLHF supports the following classic datasets, a full list of all tasks is [available here](). Uni-RLHF also supports the uploading of customizaton datasets, as long as the dataset contains `observations` and `terminals` keys.

* Install [D4RL](https://github.com/Farama-Foundation/D4RL) dependencies. Note that we made some small changes to the camera view for better visualisations.
   ```sh  
   cd d4rl
   pip install -e .
   ```
* Install [Atari](https://github.com/takuseno/d4rl-atari) dependencies.
   ```sh  
   pip install git+https://github.com/takuseno/d4rl-atari
   ```
* Install [V-D4RL](https://github.com/conglu1997/v-d4rl) dependencies. Note that v-d4rl provide image datasets and full datasets can be found on [GoogleDrive](https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI). **These must be downloaded before running the code.** And the right file structure is:  
   ```sh  
    uni_rlhf
    └───datasets
    │   └───dataset_resource
    |       └───vd4rl
    |       |   └───cheetah
    |       |   │   └───cheetah_run_medium
    |       |   │   └───cheetah_run_medium_expert
    |       |   └───humanoid
    |       |   |   |───humanoid_walk_medium
    |       |   │   └───humanoid_walk_medium_expert
    |       |   └───walker
    |       |       |───walker_walk_medium
    |       |       └───walker_walk_medium_expert
    |       └───smarts
    |          └───cruise
    |          └───curin
    |          └───left_c
    └───vue_part
    │   ...
    └───controllers
    │   ...
   ```
* Install [MiniGrid](https://github.com/Farama-Foundation/Minigrid) dependencies. There are the same dependencies as the D4RL datasets.  
* Install [SMARTS](https://github.com/huawei-noah/SMARTS/tree/master) dependencies. We employed online reinforcement learning algorithms to train two agents for datasets collection, each designed specifically for the respective scenario. The first agent demonstrates medium driving proficiency, achieving a success rate ranging from 40% to 80% in its designated scenario. In contrast, the second agent exhibits expert-level performance, attaining a success rate of 95% or higher in the same scenario. For dataset construction, 800 driving trajectories were collected using the intermediate agent, while an additional 200 were gathered via
the expert agent. By integrating the two datasets, **we compiled a mixed dataset encompassing 1,000 driving trajectories.** We upload full datasets containing image (for rendering) and vector (for training) on [GoogleDrive](https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI). **These must be downloaded before running the code.** And the right file structure is the same as v-d4rl dataset.
* Upload customization datasets. The customization datasets must be `h5df` format and contain `observations` and `terminal` keys:

    ```sh
    observations: An N by observation dimensional array of observations.
    terminals: An N dimensional array of episode termination flags. 
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- #### Docker

We also provide a Dockerfile for easy installation. You can build the docker image by running

   ```sh
   cd docker && docker build . -t <user>/uni-rlhf:0.1.0
   ``` -->

### Setup

#### MySQL
   ```
   sudo systemctl start mysql.service
   mysql -u [user] -p
   ```

#### Redis
Open another terminal and enter the following:
   ```
   sudo service redis-server stop
   redis-server
   ```

#### Running the App
Now you can run the app from the base directory with:        

   ```python3 
   conda activate rlhf
   python run.py
   ```
The app is running at: 
   ```python3 
   http://localhost:8503
   ```
You can kill all relative processes with:
   ```python3 
   python scripts/kill_process.py
   ```