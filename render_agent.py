import os
import torch
import sys

from framework_utils.Insight_Renderer import InsightRenderer
from framework_utils.NeuralAgentRenderer import NeuralAgentRenderer

sys.path.append(os.path.join(os.getcwd(), "ns_policies", "SCoBOts_framework"))
sys.path.append(os.path.join(os.getcwd(), "ns_policies", "blendrl"))

a = sys.path

from framework_utils.parser import render_parser


def main():
    parser_args = render_parser()
    agent_name = parser_args["agent"]
    agent_path = parser_args["agent_path"]
    env_name = parser_args["game"]
    fps = parser_args["fps"]
    lst_panes = parser_args["lst_panes"]
    seed = parser_args["seed"]
    print_rewards = parser_args["print_reward"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    screenshot_path = ""
    render_panes = lst_panes is not None

    if render_panes and len(lst_panes) == 1 and "heat_map" in lst_panes:
        render_panes = False

    if agent_name == "scobots":
        from framework_utils.ScoBotsRenderer import ScoBotsRenderer
        renderer = ScoBotsRenderer(agent_path=agent_path,
                               env_name=env_name,
                               fps=fps,
                               device=device,
                               screenshot_path=screenshot_path,
                               print_rewards=print_rewards,
                               render_panes=render_panes,
                               lst_panes=lst_panes,
                               seed=seed,
                               parser_args=parser_args)
    elif agent_name == "blendrl":
        from framework_utils.BlendRLRenderer import BlendRLRenderer
        deterministic = True
        renderer = BlendRLRenderer(agent_path=agent_path,
                               env_name=env_name,
                               fps=fps,
                               device=device,
                               screenshot_path=screenshot_path,
                               print_rewards=print_rewards,
                               deterministic=deterministic,
                               render_panes=render_panes,
                               lst_panes=lst_panes,
                               seed=seed,
                               env_kwargs=dict(render_oc_overlay=True))
    elif agent_name == "nudge":
        from framework_utils.BlendRLRenderer import BlendRLRenderer
        deterministic = True
        renderer = BlendRLRenderer(agent_path=agent_path,
                               env_name=env_name,
                               fps=fps,
                               device=device,
                               screenshot_path=screenshot_path,
                               print_rewards=print_rewards,
                               deterministic=deterministic,
                               render_panes=render_panes,
                               lst_panes=lst_panes,
                               seed=seed,
                               env_kwargs=dict(render_oc_overlay=True))
    elif agent_name == "neural":
        renderer = NeuralAgentRenderer(agent_path=agent_path,
                            env_name=env_name,
                            fps=fps,
                            device=device,
                            screenshot_path=screenshot_path,
                            print_rewards=print_rewards,
                            render_panes=render_panes,
                            lst_panes=lst_panes,
                            seed=seed,
                            parser_args=parser_args,
                            dopamine_pooling=parser_args["dopamine_pooling"],)
    elif agent_name == "insight":
        renderer = InsightRenderer(agent_path=agent_path,
                                   env_name=env_name,
                                   fps = fps,
                                   device = device,
                                   screenshot_path=screenshot_path,
                                   print_rewards=print_rewards,
                                   render_panes=render_panes,
                                   lst_panes=lst_panes,
                                   parser_args=parser_args,
        )
    else:
        raise NameError(f"Unknown agent {agent_name}")
    renderer.run()


if __name__ == '__main__':
    main()