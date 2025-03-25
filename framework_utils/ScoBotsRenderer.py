import logging
import os
import time
import warnings

import numpy as np
import pygame
import gymnasium as gym

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from joblib import load

from framework_utils.BaseRenderer import BaseRenderer
from ns_policies.SCoBOts_framework.scobi.core import Environment
from ns_policies.SCoBOts_framework.utils.parser.parser import get_highest_version
from ns_policies.SCoBOts_framework.viper_extract import DTClassifierModel

try:
    from pygame_screen_record import ScreenRecorder

    _screen_recorder_imported = True
except ImportError as imp_err:
    _screen_recorder_imported = False


class ScoBotsRenderer(BaseRenderer):
    """
    Renderer class for SCoBots agents, handling visualization, recording, and interaction.
    """

    def __init__(self,
                 agent_path=None,
                 env_name="seaquest",
                 fps: int = 15,
                 device="cpu",
                 screenshot_path="",
                 print_rewards=False,
                 render_panes=True,
                 lst_panes=None,
                 seed=0,
                 parser_args: dict = None):
        """
        Initializes the ScoBotsRenderer with given parameters.

        :param agent_path: Path to the trained agent model.
        :param env_name: Name of the environment.
        :param fps: Frames per second.
        :param device: Computation device (e.g., "cpu" or "cuda", default: "cpu").
        :param screenshot_path: Path to save screenshots.
        :param print_rewards: Flag to print rewards during rendering.
        :param render_panes: Whether to render additional information panes (default: True).
        :param lst_panes: List of panes to display in the UI (default: None).
        :param seed: Random seed for environment (default: 0).
        :param parser_args: Dictionary of parser arguments.
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

        version = int(parser_args["version"])
        exp_name = parser_args["exp_name"]
        variant = parser_args["variant"]
        env_str = parser_args["env_str"]
        pruned_ff_name = parser_args["pruned_ff_name"]
        hide_properties = parser_args["hide_properties"]
        viper = parser_args["viper"]
        self.record = parser_args["record"]
        nb_frames = parser_args["nb_frames"]
        self.print_rewards = parser_args["print_reward"]

        #################################################################################
        # LOAD POLICY
        if viper:
            print("loading viper tree of " + exp_name)
            if isinstance(viper, str):
                self.model = self._load_viper(viper, True)
            else:
                self.model = self._load_viper(exp_name, False)
        else:
            self.model = PPO.load(self.agent_path)

        ################################################################################
        # LOAD ENVIRONMENT
        if version == -1:
            version = get_highest_version(exp_name)
        elif version == 0:
            version = ""

        exp_name += str(version)
        vecnorm_str = "best_vecnormalize.pkl"
        vecnorm_path = Path("./ns_policies/SCoBOts_framework/resources/checkpoints", exp_name, vecnorm_str)
        self.path = Path("./ns_policies/SCoBOts_framework/resources/checkpoints", exp_name)

        # if not self._ensure_completeness(ff_file_path):
        #    print("The folder " + str(ff_file_path) + " does not contain a completed training checkpoint.")
        #    return
        if variant == "rgb":
            self.envs = make_vec_env(env_str, seed=self.seed, wrapper_class=WarpFrame)
        else:
            self.envs = Environment(env_str,
                                    focus_dir=self.path,
                                    focus_file=pruned_ff_name,
                                    hide_properties=hide_properties,
                                    draw_features=True,  # implement feature attribution
                                    reward=0)  # env reward only for evaluation

            _, _ = self.envs.reset(seed=self.seed)
            dummy_vecenv = DummyVecEnv([lambda: self.envs])
            self.envs = VecNormalize.load(vecnorm_path, dummy_vecenv)
            self.envs.training = False
            self.envs.norm_reward = False
        self.envs.reset()

        if hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
            self.env = self.envs.venv.envs[0]
            self.rgb_agent = False
        elif hasattr(self.envs, 'envs'):
            self.env = self.envs.envs[0]  # Handles cases where envs are in a DummyVecEnv or similar
            self.rgb_agent = True
        else:
            self.env = self.envs
            self.rgb_agent = True

        self.current_frame = self._get_current_frame()
        self._init_pygame(self.current_frame)

        if not self.rgb_agent:
            self.keys2actions = self.env.oc_env.unwrapped.get_keys_to_action()
            self.action_meanings = self.env.oc_env.unwrapped.get_action_meanings()

        self.ram_grid_anchor_left = self.env_render_shape[0] + 28
        self.ram_grid_anchor_top = 28

        self._recording = False

        if self.record:
            if _screen_recorder_imported:
                self._screen_recorder = ScreenRecorder(60)
                self._screen_recorder.start_rec()
                self._recording = True
                self.nb_frames = nb_frames
            else:
                print("Screen recording not available. Please install the pygame_screen_record package.")
                exit(1)
        else:
            self.nb_frames = np.inf

    def _load_viper(self, exp_name, path_provided):
        """
        Helper function to load from a dt and not a checkpoint directly
        :param exp_name:  name of the experiment
        :param path_provided:  Whether the path is provided or not
        :return: wrapped VIPER model
        """
        if path_provided:
            viper_path = Path(exp_name)
            model = load(sorted(viper_path.glob("*_best.viper"))[0])
        else:
            viper_path = Path("resources/viper_extracts/extract_output", exp_name + "-extraction")
            model = load(sorted(viper_path.glob("*_best.viper"))[0])

        wrapped = DTClassifierModel(model)

        return wrapped

    def _ensure_completeness(self, path):
        """
        Helper function ensuring that a checkpoint has completed training
        :param path: path to checkpoint.
        :return:
        """
        checkpoint = path / "best_model.zip"
        return checkpoint.is_file()

    def _get_current_frame(self):
        if self.rgb_agent:
            return self.env.render()
        else:
            return self.env.obj_obs

    def run(self):
        self.running = True
        obs = self.envs.reset()
        i = 1
        while self.running:
            self._handle_user_input()
            self.env.oc_env.render_oc_overlay = self.overlay
            if not self.paused:

                if self.human_playing:  # human plays game manually
                    action = [self._get_action()]
                    time.sleep(0.05)
                else:  # AI plays game
                    action, _ = self.model.predict(obs, deterministic=True)
                self.action = action[0]

                obs, rew, done, infos = self.envs.step(action)
                self.env.sco_obs = obs
                self.current_frame = self._get_current_frame()

                if done or self.reset:
                    if "episode" in infos and self.print_rewards:
                        episode_return = infos["episode"]["r"]
                        episode_length = infos["episode"]["l"]
                        print("episodic return:", episode_return)
                        print("episodic length:", episode_length)
                    obs= self.env.reset()

                if self.print_rewards and rew:
                    print(f"reward: {rew[0]}")

                if done:
                    if self._recording and self.nb_frames == 0:
                        self._save_recording()
                    obs = self.envs.reset()
                elif self._recording and i == self.nb_frames:
                    self._save_recording()
            self._render()

        pygame.quit()

    def _save_recording(self):
        """
        Save the current frame recording.
        """
        filename = Path.joinpath(self.path, "recordings")
        filename.mkdir(parents=True, exist_ok=True)
        self._screen_recorder.stop_rec()  # stop recording
        print(self.env.spec.name)
        if self.rgb_agent:
            filename = Path.joinpath(filename, f"{self.env.spec.name}.avi")
        else:
            filename = Path.joinpath(filename, f"{self.env.oc_env.game_name}.avi")
        i = 0
        while os.path.exists(filename):
            i += 1
            if self.rgb_agent:
                filename = Path.joinpath(self.path, "recordings", f"{self.env.spec.name}_{i}.avi")
            else:
                filename = Path.joinpath(self.path, "recordings", f"{self.env.oc_env.game_name}_{i}.avi")
        print(filename)
        self._screen_recorder.save_recording(filename)
        print(f"Recording saved as {filename}")
        self._recording = False

    def _render(self, frame=None):
        lst_possible_panes = ["selected_actions", "semantic_actions", "state_usage"]

        self.window.fill((0, 0, 0))  # clear the entire window
        self._render_env()

        anchor = (self.env_render_shape[0] + 10, 25)

        panes_row = []

        # Render selected actions
        if "selected_actions" in self.lst_panes:
            pane_size = self._render_selected_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + pane_size[1] + 10)
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        # Render semantic actions
        if "semantic_actions" in self.lst_panes:
            pane_size = self._render_semantic_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        # render rgb states
        if "state_usage" in self.lst_panes:
            pane_size = self.render_state_usage(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        # Warning for requesting not implemented panes for ScoBots
        remains = [pane for pane in self.lst_panes if pane not in lst_possible_panes]
        if remains and not self.pane_warning:
            self.pane_warning = True
            logging.basicConfig(level=logging.WARNING)
            logging.warning(f"No panes available for {remains} in SCoBots! Possible panes are: {lst_possible_panes}")

        pygame.display.flip()
        pygame.event.pump()

        if not self.fast_forward:
            self.clock.tick(self.fps)
