# Framework
## Prerequisites
1. Use Python 3.10
2. Clone the Repository
3. cd into the Repository and init all submodules
``` git submodule update --init ```
4. Checkout either master or main for all submodules
   1. ``` ns_policies/blendrl ``` main
   2. ``` ns_policies/SCoBOts_framework ``` master
   3. ``` ns_policies/insight_oc ``` main
   4. ``` object_extraction/HackAtari ``` master
   5. ``` object_extraction/OC_Atari_framework ``` master
## Setup
```
pip install -r requirements.txt && pip install "stable-baselines3[extras]==2.0.0"
```
1. Inside ```ns_policies/blendrl/nsfr/``` run
    ```
    python setup.py develop
    ```
2. Inside ```ns_policies/blendrl/nudge/``` run
    ```
    python setup.py develop
    ```
3. Inside ```ns_policies/blendrl/neumann/``` run
    ```
    python setup.py develop
    ```
Install the other dependencies:
```
   pip install torch==2.5.1 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch_geometric
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
   pip install vidmaker 
   pip install "gymnasium[atari, accept-rom-license]"
   ```
 
### SCoBOts agents
Inside ``` ns_policies/SCoBOts_framework ``` run the following commands to get pretrained SCoBOts agents:
 
```
# Download the agents (only seed0)
wget https://hessenbox.tu-darmstadt.de/dl/fi47F21YBzVZBRfGPKswumb7/resources_seed0.zip
unzip resources_seed0.zip
```
or
```
# Download the agents (all seeds)
wget https://hessenbox.tu-darmstadt.de/dl/fiPLH36Zwi8EVv8JaLU4HpE2/resources_all.zip
unzip resources_all.zip
```

### BlendRL agents
Inside ``` ns_policies/blendrl ``` run the following commands to get pretrained BlendRL agents:
```
wget https://hessenbox.tu-darmstadt.de/dl/fiCNznPuWkALH8JaCJWHeeAV/models.zip
unzip models.zip
rm models.zip
```
 
### INSIGHT agents
Inside ``` ns_policies/insight_oc ``` run the following commands to get a dummy INSIGHT agent:
```
mkdir models
cd models
wget https://hessenbox.tu-darmstadt.de/dl/fiXx4TFMfQZAzhfbffgdm8cz/Pong_AgentSimplified_final.pth
```
 
### Neural agents
Download deep RL agents from [Google Drive](https://drive.google.com/drive/folders/1-6l2A82dGlBZ52jlKEuo9vTCdOcfFZHJ?usp=sharing) and save them inside ``` ns_policies/neural/agents/<GAME> ```.
 
### NUDGE agents
Download nudge agents from the blendrl github branches for nudge [Example link](https://github.com/ml-research/blendrl/tree/Freeway/out_freeway/runs/freeway_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0)
and save them inside ``` ns_policies/blendrl/out/ ```.
 
**Note**: For NUDGE agents to work, exchange the ``` ns_policies/blendrl/in/envs/<GAME> ``` folder for the <GAME> folder used in the branch implementation and change the wrappers 
``gym.wrappers.Autoreset, gym.wrappers.GrayscaleObservation, gym.wrappers.FrameStackObservation`` for the corresponding wrappers in gym v0.28.1 
``gym.wrappers.AutoResetWrapper, gym.wrappers.GrayScaleObservation, gym.wrappers.FrameStack `` in ``env.py`` and ``env_vectorized.py``.
 
## Run the game using an agent
 
```
python render_agent.py -a METHOD -g GAME -pl PANES_LIST -ap AGENT_PATH
```
 
* ```METHOD``` is the type of agent you want to use (e.g. scobots, blendrl)
* ```GAME``` is the type of game you want to play (e.g. Pong, seaquest)
* ```PANES_LIST``` list of panes that will be displayed next to the game (e.g. selected_actions, policy)
* ```AGENT_PATH``` path to your local agent
 
## Examples
SCoBOts agent downloaded from above playing pong: 
``` 
python render_agent.py -a scobots -g Pong -pl selected_actions -ap ./ns_policies/SCoBOts_framework/resources/checkpoints/Pong_seed0_reward-human_oc_pruned/best_model.zip
```
blendrl agent downloaded from above playing kangaroo: 
``` 
python render_agent.py -a blendrl -ap ./ns_policies/blendrl/models/kangaroo_demo -g kangaroo
```
Insight example
```
python render_agent.py -a insight -g Pong -pl selected_actions semantic_actions -ap AGENT_PATH --print-reward
```
Neural Agent example
```
python render_agent.py -a neural -g Pong -pl selected_actions semantic_actions heat_map -ap AGENT_PATH --print-reward
```
NUDGE example
```
python render_agent.py -a nudge -g Freeway -pl selected_actions semantic_actions -ap AGENT_PATH --print-reward
