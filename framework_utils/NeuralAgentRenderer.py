from stable_baselines3 import PPO

from framework_utils.BaseRenderer import BaseRenderer
from object_extraction.OC_Atari_framework.ocatari import OCAtari


class NeuralAgentRenderer(BaseRenderer):

    def __init__(self, agent_path=None, env_name="seaquest", fps: int = 15, device="cpu", screenshot_path="",
                 render_panes=True, seed=0):

        super().__init__(agent_path, env_name, fps, device, screenshot_path, render_panes, seed)

        # load model
        self.model = PPO.load(agent_path)

        # load environment
        self.env = OCAtari(env_name=self.env_name, mode='ram', hud=False, obs_mode="dqn", render_oc_overlay=True)
        self.env.reset()

        self.current_frame = self._get_current_frame()
        self._init_pygame(self.current_frame)

    def _get_current_frame(self):
        return self.env.obj_obs