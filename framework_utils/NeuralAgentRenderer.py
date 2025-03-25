import logging
import os
import sys
import time
import warnings
import pygame
import torch

from framework_utils.BaseRenderer import BaseRenderer

import object_extraction.OC_Atari_framework.saliency as saliency

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
                 print_rewards=False,
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
                         print_rewards=print_rewards,
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
            self.env.seed = seed
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
        self.model, self.policy = load_agent(agent_path, self.env, self.device)

        #################################################################################
        # RENDERER INITIALIZATION
        self.current_frame = self._get_current_frame()
        self._init_pygame(self.current_frame)

        self.keys2actions = self.env.unwrapped.get_keys_to_action()
        self.action_meanings = self.env.unwrapped.get_action_meanings()

        self.history = {'ins': [], 'obs': []}  # For heat map

    def _get_current_frame(self):
        return self.env.render()


    def run(self):
        self.running = True
        obs, _ = self.env.reset()
        self.heat_counter = -1
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
                    action = self.policy(obs)[0]
                if torch.is_tensor(action):
                    self.action = action.unsqueeze(0)
                else:
                    self.action = [action]

                obs, rew, terminated, truncated, info  = self.env.step(action)

                self.action = action

                if "heat_map" in self.lst_panes:

                    self.og_obs = obs

                    self.heat_counter += 1

                    self.update_history()

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
        lst_possible_panes = ["selected_actions", "semantic_actions", "heat_map"]

        self.window.fill((0, 0, 0))  # clear the entire window
        if "heat_map" in self.lst_panes:
            self._render_heat_map()
        else:
            self._render_env()

        anchor = (self.env_render_shape[0] + 10, 25)

        panes_row = []

        if "selected_actions" in self.lst_panes:
            pane_size = self._render_selected_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + pane_size[1] + 10)
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        if "semantic_actions" in self.lst_panes:
            pane_size = self._render_semantic_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        remains = [pane for pane in self.lst_panes if pane not in lst_possible_panes]
        if remains and not self.pane_warning:
            self.pane_warning = True
            logging.basicConfig(level=logging.WARNING)
            logging.warning(f"No panes available for {remains} in neural agents! Possible panes are: {lst_possible_panes}")

        pygame.display.flip()
        pygame.event.pump()

        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_heat_map(self, density=5, radius=5, prefix='default'):
        '''
        Render the heatmap on top of the game.
        '''
        # Render normal game frame.
        if len(self.history['ins']) <= 1:
            self._render_env()

        # Render game frame with heat map.
        elif len(self.history['ins']) > 1:
            radius, density = 5, 5
            upscale_factor = 5
            actor_saliency = saliency.score_frame(
                self.env, self.model, self.history, self.heat_counter, radius, density, interp_func=saliency.occlude,
                mode="actor"
            )  # shape (84,84)

            critic_saliency = saliency.score_frame(
                self.env, self.model, self.history, self.heat_counter, radius, density, interp_func=saliency.occlude,
                mode="critic"
            )  # shape (84,84)

            frame = self.history['ins'][
                self.heat_counter].squeeze().copy()  # Get the latest frame with shape (210,160,3)

            frame = saliency.saliency_on_atari_frame(actor_saliency, frame, fudge_factor=400, channel=2)
            frame = saliency.saliency_on_atari_frame(critic_saliency, frame, fudge_factor=600, channel=0)

            frame = frame.swapaxes(0, 1).repeat(upscale_factor, axis=0).repeat(upscale_factor, axis=1)  # frame has shape (210,160,3), upscale to (800,1050,3). From ocatari/core.py/render()

            heat_surface = pygame.Surface(self.env_render_shape)

            pygame.pixelcopy.array_to_surface(heat_surface, frame)

            self.window.blit(heat_surface, (0, 0))

    def update_history(self):
        '''
        Method for updating the history of the game. Needed for the heatmap.
        '''
        self.history['ins'].append(self.env._env.render())  # Original rgb observation with shape (210,160,3)
        self.history['obs'].append(self.og_obs)  # shape (4,84,84), no prepro necessary