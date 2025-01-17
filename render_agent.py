import torch
import sys

sys.path.append("./SCoBots_framework")

from ns_policies.SCoBOts_framework.utils.parser.parser import render_parser
from utils.ScoBotsRenderer import ScoBotsRenderer


def main():
    agent_path = "ns_policies/SCoBOts_framework/resources/checkpoints/Pong_seed0_reward-human_oc_pruned",
    env_name = "Pong",
    fps = 15,
    device = "cuda:0" if torch.cuda.is_available() else "cpu",
    screenshot_path = "",
    render_panes = True,
    parser_args = render_parser()
    renderer = ScoBotsRenderer(agent_path, env_name, fps, device, screenshot_path, render_panes, parser_args)
    renderer.run()


if __name__ == '__main__':
    main()