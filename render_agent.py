import os
import torch
import sys


sys.path.append(os.path.join(os.getcwd(), "ns_policies\SCoBOts_framework"))
sys.path.append(os.path.join(os.getcwd(), r"ns_policies\blendrl"))

a = sys.path

from framework_utils.BlendRLRenderer import BlendRLRenderer
from ns_policies.SCoBOts_framework.utils.parser.parser import render_parser
from framework_utils.ScoBotsRenderer import ScoBotsRenderer


def main():
    # agent_path = "./ns_policies/SCoBOts_framework/resources/checkpoints/Pong_seed0_reward-human_oc_pruned/best_model.zip"
    # env_name = "Pong"
    # fps = 15
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # screenshot_path = ""
    # render_panes = True
    # seed = 0
    # parser_args = render_parser()
    # renderer = ScoBotsRenderer(agent_path, env_name, fps, device, screenshot_path, render_panes, seed, parser_args)
    # renderer.run()

    agent_path = "./ns_policies/blendrl/out/runs/seaquest_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_5_steps_128_pretrained_False_joint_True_0"
    env_name = "seaquest"
    fps= 5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    screenshot_path = ""
    deterministic = True
    render_panes = True
    seed = 0
    renderer = BlendRLRenderer(agent_path=agent_path,
                               env_name=env_name,
                               fps=fps,
                               device=device,
                               screenshot_path=screenshot_path,
                               deterministic=deterministic,
                               render_panes=render_panes,
                               seed=seed,
                               env_kwargs=dict(render_oc_overlay=True))
    renderer.run()


if __name__ == '__main__':
    main()