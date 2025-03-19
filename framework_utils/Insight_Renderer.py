
import os
import time

import pygame
import torch
import sys
import logging

from stable_baselines3.common.monitor import Monitor

from ns_policies.insight_oc.cleanrl.hackatari_env import HackAtariWrapper
from ns_policies.insight_oc.cleanrl.hackatari_utils import get_reward_func_path

sys.path.insert(1,os.path.join(os.getcwd(), "ns_policies", "insight_oc", "cleanrl"))


from framework_utils.BaseRenderer import BaseRenderer


def make_env(env_id, seed, rewardfunc_path, modifs):
    def thunk():
        env = HackAtariWrapper(env_id, modifs=modifs, rewardfunc_path=rewardfunc_path)
        env.seed(seed)
        env = Monitor(env)
        return env

    return thunk

class InsightRenderer(BaseRenderer):
    """
    Renderer class for neural agents, handling visualization, recording, and interaction.
    """

    def __init__(self, agent_path=None,
                 env_name="Pong",
                 fps: int = 15,
                 device="cpu",
                 screenshot_path="",
                 print_rewards=False,
                 render_panes=True,
                 lst_panes=None,
                 seed=0,
                 parser_args=None):
        """
        :param agent_path: file path to the agent weights
        :param env_name: name of the environment
        :param fps: frames per second
        :param device: device to use
        :param screenshot_path: path to save screenshots
        :param print_rewards: whether to print rewards
        :param render_panes: whether to render panes, or not
        :param seed: random seed
        :param parser_args: arguments passed to parser
        """
        super().__init__(agent_path=agent_path,
                         env_name=env_name,
                         fps=fps,
                         device=device,
                         screenshot_path=screenshot_path,
                         print_rewards=print_rewards,
                         render_panes=render_panes,
                         lst_panes=lst_panes,
                         seed=seed)

        ################################################################################
        # LOAD ENVIRONMENT
        rewardfunc_path = get_reward_func_path(env_name, parser_args["reward_function"]) if parser_args["reward_function"] else None
        self.env = HackAtariWrapper(env_name, modifs=parser_args["modifs"], rewardfunc_path=rewardfunc_path)
        self.env.seed(seed)
        self.env = Monitor(self.env)
        self.env.reset()

        #################################################################################
        # LOAD POLICY
        self.model = torch.load(agent_path, weights_only=False, map_location=device)
        self.model.network.eval()
        self.threshold = parser_args["threshold"]

        #################################################################################
        # RENDERER INITIALIZATION
        self.current_frame = self._get_current_frame()
        self._init_pygame(self.current_frame)


        self.keys2actions = self.env.env.unwrapped.get_keys_to_action()
        self.action_meanings = self.env.env.unwrapped.get_action_meanings()


    def _get_current_frame(self):
        return self.env.env.render()


    def run(self):
        self.running = True

        obs = self.env.reset()
        if isinstance(obs, tuple):  # Handle vectorized envs that return (obs, info)
            obs = obs[0]

        action_func = lambda t: self.model.get_action_and_value(t, actor="eql")[0]
        while self.running:
            self._handle_user_input()
            self.env.render_oc_overlay = self.overlay

            if not self.paused:

                if self.human_playing:  # human plays game manually
                    action = self._get_action()
                    time.sleep(0.05)
                else:
                    obs_tensor = torch.Tensor(obs).to(self.device)
                    action = action_func(obs_tensor.unsqueeze(0))
                self.action = action

                obs, rew, terminated, truncated, info  = self.env.step(action.cpu().numpy().item())

                self.current_frame = self._get_current_frame()

                if terminated or truncated or self.reset:
                    if "episode" in info and self.print_rewards:
                        episode_return = info["episode"]["r"]
                        episode_length = info["episode"]["l"]
                        print("episodic return:", episode_return)
                        print("episodic length:", episode_length)
                    obs, _ = self.env.reset()

                if self.print_rewards and rew:
                    print(f"reward: {rew}")

            self._render()

        pygame.quit()


    def _render(self, frame=None):
        lst_possible_panes = ["selected_actions", "semantic_actions"]

        self.window.fill((0, 0, 0))  # clear the entire window
        self._render_env()

        anchor = (self.env_render_shape[0] + 10, 25)

        if "selected_actions" in self.lst_panes:
            pane_size = self._render_selected_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + pane_size[1])
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])

        if "semantic_actions" in self.lst_panes:
            pane_size = self._render_semantic_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + pane_size[1])
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])

        remains = [pane for pane in self.lst_panes if pane not in lst_possible_panes]
        if remains and not self.pane_warning:
            self.pane_warning = True
            logging.basicConfig(level=logging.WARNING)
            logging.warning(f"No panes available for {remains} in Insight! Possible panes are: {lst_possible_panes}")

        pygame.display.flip()
        pygame.event.pump()

        if not self.fast_forward:
            self.clock.tick(self.fps)
