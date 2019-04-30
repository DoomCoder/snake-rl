## Python modules allowing to train DQN agent, observe agent's games and collect its stats.

#### Report: https://www.overleaf.com/read/khmgdtdxtmhk

### main.py
Train convolutional selected model on environment.  
Currently doesn't support arguments.

### test_agent_play.py
#### Supported arguments:  
- -a, --agent  
 Supported agents: 
   - tDQN - Deep Q Learning in Pytorch
   - tAC - Actor Critic in Pytorch
   - kDQN - Deep Q Learning in Keras  
- --model [PATH] - path to saved model's weights  
 Agents currently supporting default weights:
   - tAC

Watch trained agent play snake or collect stats about agent's performance  

### Installation
This project requires snake_gym repo to be present in the same directory as this one.  
https://github.com/Gizzio/snake_gym  
The tree should look like this:
- your_dir
  - snake_gym
  - snake-rl  
  
For example:
```angular2html
mkdir snake
cd snake
git clone git@github.com:DoomCoder/snake-rl.git
git clone git@github.com:Gizzio/snake_gym.git
```
1. Setup directories as descibed above
2. Install requirements.txt from snake_gym
3. Install requirementx.txt from snake-rl  
If your machine doesn't support CUDA, use requirements_cpu.txt
4. Done!
