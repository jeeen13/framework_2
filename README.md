# NSRL
End to end Neuro-Symbolic Reinforcement Learning

# Description
This frameworks gathers and compares many different object-centric/neurosymbolic RL methods.
Agents can be run using different object extractions methods:
* [OCAtari (baseline)](https://github.com/k4ntz/OC_Atari) [1]
* [SPOC]() [2]
* [MOC]() [2]
* [INSIGHT OCNN](https://github.com/k4ntz/SCoBots) [3]

as well as different interpretable policies:
* [INSIGHT]() [3]
* [SCoBots]() [4]
* [NUDGE](https://github.com/ml-research/blendrl) [5]
* [BlendRL](https://github.com/ml-research/blendrl) [6]

# Install
We advise you to install the dependencies from within dependencies folder

```bash
cd dependencies
git clone https://github.com/k4ntz/OC_Atari
cd OC_Atari
pip install -e .
```

You can do this with all the referenced baselines (clicable links above). 

# Scripts

## Object detectors
To visualize the detected objects: 
```bash
python3 scripts/run_detector.py -d DETECTOR -g GAME 
```
* if `DETECTOR` not provided, it uses the RAM extraction of OCAtari. 
 else use `SPOC`, `SLOC` or `INS_CNN` to use the corresponding object detector.  
* `GAME` should be an ATARI game, covered by OCAtari.

To evaluate the performances of a detection method on a specific games:
```bash
python3 scripts/eval_detector.py -d DETECTOR -g GAME 
```

## Policies 
### Prerequisites

1. Use Python 3.10
2. Clone the Repository
3. Checkout the policy branch
4. cd into the Repository and init all submodules 
``` git submodule update --init ```
5. Checkout either master or main for all submodules
   1. ``` nsrl/policies/blendrl ``` nsrl_framework
   2. ``` nsrl/policies/scobots ``` nsrl_framework
   3. ``` nsrl/policies/insight ``` main
   4. ``` nsrl/policies/hackatari ``` master
   5. ``` nsrl/policies/ocatari ``` master
### Setup
create venv ``` python -m venv env ``` and activate it then follow the following instructions
```
pip install -U pip && pip install -r requirements.txt && pip install "stable-baselines3[extras]==2.0.0"
```
1. Inside ```nsrl/policies/blendrl/nsfr/``` run
    ```
    python setup.py develop
    ```
2. Inside ```nsrl/policies/blendrl/nudge/``` run
    ```
    python setup.py develop
    ```
3. Inside ```nsrl/policies/blendrl/neumann/``` run
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
   
#### SCoBOts agents
Inside ``` nsrl/policies/scobots ``` run the following commands to get pretrained SCoBOts agents:

```
# Download the agents (only seed0)
wget https://hessenbox.tu-darmstadt.de/dl/fi47F21YBzVZBRfGPKswumb7/resources_seed0.zip
unzip resources_seed0.zip
rm resources_seed0.zip
```
or
```
# Download the agents (all seeds)
wget https://hessenbox.tu-darmstadt.de/dl/fiPLH36Zwi8EVv8JaLU4HpE2/resources_all.zip
unzip resources_all.zip
rm resources_all.zip
```

#### BlendRL agents
Inside ``` nsrl/policies/blendrl ``` run the following commands to get pretrained BlendRL agents:
```
wget https://hessenbox.tu-darmstadt.de/dl/fiCNznPuWkALH8JaCJWHeeAV/models.zip
unzip models.zip
rm models.zip
```

#### INSIGHT agents
Inside ``` nsrl/policies/insight ``` run the following commands to get a dummy INSIGHT agent:
```
mkdir models
cd models
wget https://hessenbox.tu-darmstadt.de/dl/fiXx4TFMfQZAzhfbffgdm8cz/Pong_AgentSimplified_final.pth
```

#### Neural agents
Download deep RL agents from [Google Drive](https://drive.google.com/drive/folders/1-6l2A82dGlBZ52jlKEuo9vTCdOcfFZHJ?usp=sharing) and save them inside ``` nsrl/policies/neural/agents/<GAME> ```.

#### NUDGE agents
Download nudge agents from the blendrl github branches for nudge [Example link](https://github.com/ml-research/blendrl/tree/Freeway/out_freeway/runs/freeway_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0)
and save them inside ``` nsrl/policies/blendrl/out/ ```.

**Note**: For NUDGE agents to work, exchange the ``` nsrl/policies/blendrl/in/envs/<GAME> ``` folder for the <GAME> folder used in the branch implementation and change the wrappers 
``gym.wrappers.Autoreset, gym.wrappers.GrayscaleObservation, gym.wrappers.FrameStackObservation`` for the corresponding wrappers in gym v0.28.1 
``gym.wrappers.AutoResetWrapper, gym.wrappers.GrayScaleObservation, gym.wrappers.FrameStack `` in ``env.py`` and ``env_vectorized.py``.

### Run the game using an agent

```
python render_agent.py -a METHOD -g GAME -pl PANES_LIST -ap AGENT_PATH
```

* ```METHOD``` is the type of agent you want to use (e.g. scobots, blendrl)
* ```GAME``` is the type of game you want to play (e.g. Pong, seaquest)
* ```PANES_LIST``` list of panes that will be displayed next to the game (e.g. selected_actions, policy)
* ```AGENT_PATH``` path to your local agent
  
### Examples
SCoBOts agent downloaded from above playing pong: 
``` 
python scripts/render_agent.py -a scobots -g Pong -pl selected_actions -ap ./nsrl/policies/scobots/resources/checkpoints/Pong_seed0_reward-human_oc_pruned/best_model.zip
```
blendrl agent downloaded from above playing kangaroo: 
``` 
python scripts/render_agent.py -a blendrl -ap ./nsrl/policies/blendrl/models/kangaroo_demo -pl heat_map logic_valuations selected_actions semantic_actions policy state_usage logic_action_rules -g kangaroo
```
Insight example
```
python scripts/render_agent.py -a insight -g Pong -pl selected_actions semantic_actions -ap AGENT_PATH --print-reward
```
Neural Agent example
```
python scripts/render_agent.py -a neural -g Pong -pl selected_actions semantic_actions heat_map -ap AGENT_PATH --print-reward
```
NUDGE example
```
python scripts/render_agent.py -a nudge -g Freeway -pl selected_actions semantic_actions -ap AGENT_PATH --print-reward
```
# References
[1]
[2]
[3]
[4]
[5]
[6]
