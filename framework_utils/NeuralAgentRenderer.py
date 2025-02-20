import os
import sys
import time
import warnings
import pygame
import torch

from framework_utils.BaseRenderer import BaseRenderer


# add OC_Atari path
oc_atari_path = os.path.join(os.getcwd(), "object_extraction/OC_Atari_framework")
sys.path.insert(1, oc_atari_path)


class NeuralAgentRenderer(BaseRenderer):
    """
    Renderer class for neural agents, handling visualization, recording, and interaction.
    """

    def __init__(self, agent_path=None,
                 env_name="Pong",
                 fps: int = 15,
                 device="cpu",
                 screenshot_path="",
                 print_reward=False,
                 render_panes=True,
                 lst_panes=None,
                 seed=0,
                 parser_args=None,
                 dopamine_pooling=True):
        """
        :param agent_path: file path to the agent weights
        :param env_name: name of the environment
        :param fps: frames per second
        :param device: device to use
        :param screenshot_path: path to save screenshots
        :param print_reward: whether to print rewards
        :param render_panes: whether to render panes, or not
        :param seed: random seed
        :param parser_args: arguments passed to parser
        :param dopamine_pooling: whether to use dopamine frame pooling
        """
        super().__init__(agent_path=agent_path,
                         env_name=env_name,
                         fps=fps,
                         device=device,
                         screenshot_path=screenshot_path,
                         print_rewards=print_reward,
                         render_panes=render_panes,
                         lst_panes=lst_panes,
                         seed=seed)

        ################################################################################
        # LOAD ENVIRONMENT
        if parser_args["environment"] == "ocatari":
            from object_extraction.OC_Atari_framework.ocatari.core import OCAtari
            self.env = OCAtari(
                env_name=env_name,
                hud=False,
                render_mode="rgb_array",
                obs_mode="dqn",
                render_oc_overlay=False,
                frameskip=parser_args.get("frameskip", 4)
            )
            self.env.metadata['render_fps'] = self.fps
            self.env.seed(seed)
        elif parser_args["environment"] == "hackatari":
            from object_extraction.HackAtari.hackatari.core import HackAtari
            self.env = HackAtari(
                env_name,
                hud=False,
                render_mode="rgb_array",
                obs_mode="dqn",
                render_oc_overlay=True,
                dopamine_pooling=dopamine_pooling,
                frameskip=parser_args.get("frameskip", 1)
            )
        else:
            raise Exception("Unknown environment {}".format(parser_args["environment"]))
        self.env.reset()

        #################################################################################
        # LOAD POLICY
        from object_extraction.OC_Atari_framework.ocatari.utils import load_agent
        _, self.model = load_agent(agent_path, self.env, self.device)

        #################################################################################
        # RENDERER INITIALIZATION
        self.current_frame = self._get_current_frame()
        self._init_pygame(self.current_frame)

        self.keys2actions = self.env.unwrapped.get_keys_to_action()
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        self.print_reward = print_reward


    def _get_current_frame(self):
        return self.env.render()


    def run(self):
        self.running = True
        obs, _ = self.env.reset()
        while self.running:
            self._handle_user_input()
            self.env.render_oc_overlay = self.overlay

            if not self.paused:

                if self.human_playing:  # human plays game manually
                    action = self._get_action()
                    time.sleep(0.05)
                else:
                    obs = torch.Tensor(obs)#.to(self.device)
                    obs = obs.unsqueeze(0)
                    action = self.model(obs)[0]
                if torch.is_tensor(action):
                    self.action = action.unsqueeze(0)
                else:
                    self.action = [action]

                obs, rew, terminated, truncated, _  = self.env.step(action)

                self.current_frame = self._get_current_frame()

                if terminated or truncated or self.reset:
                    obs, _ = self.env.reset()

                if self.print_reward and rew:
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
        if remains:
            warnings.warn(f"No panes available for {remains} in SCoBots! Possible panes are: {lst_possible_panes}",
                          UserWarning)

        pygame.display.flip()
        pygame.event.pump()

        if not self.fast_forward:
            self.clock.tick(self.fps)
