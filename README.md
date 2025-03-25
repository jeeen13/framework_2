# Framework
## Prerequisites
1. Use Python 3.10
2. Clone the Repository
3. cd into the Repository and init all submodules
``` git submodule update --init ```
4. Checkout either master or main for all submodules
   1. ``` ns_policies/blendrl ```
   2. ``` ns_policies/SCoBOts_framework ```
   3. ``` ns_policies/insight_oc ```
   4. ``` object_extraction/HackAtari ```
   5. ``` object_extraction/OC_Atari_framework ```
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
   pip install torch==1.12 torchvision==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch_geometric
   pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12+cpu.html
   pip install vidmaker 
   pip install "gymnasium[atari, accept-rom-license]"
   ```
   
### SCoBOts agents
Inside ``` ns_policies/SCoBOts_framework ``` run the following commands to get pretrained scobots agents:

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

### blendrl agents

```
wget https://hessenbox.tu-darmstadt.de/dl/fiCNznPuWkALH8JaCJWHeeAV/models.zip
unzip models.zip
rm models.zip
```

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
