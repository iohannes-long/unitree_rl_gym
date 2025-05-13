pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd ../isaacgym/python
pip install -e .
cd examples
python 1080_balls_of_solitude.py
cd ../../..

cd rsl_rl
git checkout v1.0.2
pip install -e .
cd ..

cd unitree_rl_gym
pip install -e .

sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo apt-key add 3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

mkdir -p logs/h1
python legged_gym/scripts/play.py --task=h1 --rl_device=cpu --sim_device=cpu --num_envs=1
# change rsl_rl/rsl_rl/runners/on_policy_runner.py, loaded_dict = torch.load(path) ---> loaded_dict = torch.load(path, map_location=torch.device('cpu'))

cd deploy/deploy_mujoco
pip install mujoco
python deploy_mujoco.py h1.yaml

python station_sim.py sim.yaml