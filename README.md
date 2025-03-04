# Framework

## Setup
```
pip install -r requirements.txt && pip install "stable-baselines3[extras]==2.0.0"
```
1. Inside ```nsfr/``` run
    ```
    python setup.py develop
    ```
2. Inside ```nudge/``` run
    ```
    python setup.py develop
    ```
3. Inside ```neumann/``` run
    ```
    python setup.py develop
    ```
Install the other dependencies:
```
   pip install torch==2.5.1 torchvision==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch_geometric
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
   pip install vidmaker 
   pip install "gymnasium[atari, accept-rom-license]"
   ```
   
### SCoBOts agents

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

## Run the game using an agent

```
python render_agent.py -a METHOD -g GAME -pl PANES_LIST -ap AGENT_PATH
```

* ```METHOD``` is the type of agent you want to use (e.g. scobots, blendrl)
* ```GAME``` is the type of game you want to play (e.g. Pong, seaquest)
* ```PANES_LIST``` list of panes that will be displayed next to the game (e.g. selected_actions, policy)
* ```AGENT_PATH``` path to your local agent