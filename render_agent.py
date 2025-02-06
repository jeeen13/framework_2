import os
import torch
import sys

from framework_utils.NeuralAgentRenderer import NeuralAgentRenderer

sys.path.append(os.path.join(os.getcwd(), "ns_policies", "SCoBOts_framework"))
sys.path.append(os.path.join(os.getcwd(), "ns_policies", "blendrl"))

a = sys.path

from framework_utils.BlendRLRenderer import BlendRLRenderer
from framework_utils.parser import render_parser
from framework_utils.ScoBotsRenderer import ScoBotsRenderer


def main():
    parser_args = render_parser()
    env_name = parser_args["game"]
    agent_name = parser_args["agent"]
    fps = parser_args["fps"]
    seed = parser_args["seed"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    screenshot_path = ""
    lst_panes = parser_args["lst_panes"]
    render_panes = lst_panes is not None
    # blendrl: ./ns_policies/blendrl/out/runs/seaquest_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_5_steps_128_pretrained_False_joint_True_0"
    # scobots: ./ns_policies/SCoBOts_framework/resources/checkpoints/Pong_seed0_reward-human_oc_pruned/best_model.zip
    agent_path = parser_args["agent_path"]

    if agent_name == "scobots":
        renderer = ScoBotsRenderer(agent_path, env_name, fps, device, screenshot_path, render_panes, lst_panes, seed, parser_args)

    elif agent_name == "blendrl":
        deterministic = True
        renderer = BlendRLRenderer(agent_path=agent_path,
                               env_name=env_name,
                               fps=fps,
                               device=device,
                               screenshot_path=screenshot_path,
                               deterministic=deterministic,
                               render_panes=render_panes,
                                lst_panes=lst_panes,
                               seed=seed,
                               env_kwargs=dict(render_oc_overlay=True))
    elif agent_name == "neural":
        NeuralAgentRenderer(agent_path=agent_path,
                            env_name=env_name,
                            fps=fps,
                            device=device,
                            screenshot_path=screenshot_path,
                            render_panes=render_panes,
                            seed=seed)
    else:
        raise NameError(f"Unknown agent {agent_name}")
    renderer.run()


if __name__ == '__main__':
    main()