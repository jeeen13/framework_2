import os
import time

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
    window: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self,
                 agent_path = None,
                 env_name = "seaquest",
                 fps: int = 15,
                 device = "cpu",
                 screenshot_path = "",
                 render_panes=True,
                 seed = 0,
                 parser_args: dict = None):
        super().__init__(agent_path, env_name, fps, device, screenshot_path, render_panes, seed)

        version = int(parser_args["version"])
        exp_name = parser_args["exp_name"]
        variant = parser_args["variant"]
        env_str = parser_args["env_str"]
        pruned_ff_name = parser_args["pruned_ff_name"]
        hide_properties = parser_args["hide_properties"]
        viper = parser_args["viper"]
        self.record = parser_args["record"]
        nb_frames = parser_args["nb_frames"]
        self.print_reward = parser_args["print_reward"]

        # Load model
        if viper:
            print("loading viper tree of " + exp_name)
            if isinstance(viper, str):
                self.model = self._load_viper(viper, True)
            else:
                self.model = self._load_viper(exp_name, False)
        else:
            self.model = PPO.load(self.agent_path)

        # Load environment
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


    # Helper function to load from a dt and not a checkpoint directly
    def _load_viper(self, exp_name, path_provided):
        if path_provided:
            viper_path = Path(exp_name)
            model = load(sorted(viper_path.glob("*_best.viper"))[0])
        else:
            viper_path = Path("resources/viper_extracts/extract_output", exp_name + "-extraction")
            model = load(sorted(viper_path.glob("*_best.viper"))[0])

        wrapped = DTClassifierModel(model)

        return wrapped

    # Helper function ensuring that a checkpoint has completed training
    def _ensure_completeness(self, path):
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
                if self.human_playing:
                    action = [self._get_action()]
                    time.sleep(0.05)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                self.action = action
                obs, rew, done, infos = self.envs.step(action)
                self.env.sco_obs = obs
                self.current_frame = self._get_current_frame()
                if self.print_reward and rew[0]:
                    print(rew[0])
                if done:
                    if self._recording and self.nb_frames == 0:
                        self._save_recording()
                    obs = self.envs.reset()
                elif self._recording and i == self.nb_frames:
                    self._save_recording()
            self._render()

        pygame.quit()


    def _save_recording(self):
        filename = Path.joinpath(self.path, "recordings")
        filename.mkdir(parents=True, exist_ok=True)
        self._screen_recorder.stop_rec() # stop recording
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

    def _render(self, frame = None):
        self.window.fill((0,0,0))  # clear the entire window
        self._render_env()
        self._render_selected_action(0)
        self._render_semantic_action(0)
        pygame.display.flip()
        pygame.event.pump()

        if not self.fast_forward:
            self.clock.tick(self.fps)