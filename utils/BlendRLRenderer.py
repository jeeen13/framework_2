import pygame

from typing import Union

import torch

from BaseRenderer import BaseRenderer
from ns_policies.blendrl_framework.nudge.agents.logic_agent import NsfrActorCritic
from ns_policies.blendrl_framework.nudge.agents.neural_agent import ActorCritic
from ns_policies.blendrl_framework.nudge.env import NudgeBaseEnv
from ns_policies.blendrl_framework.nudge.utils import load_model, yellow


class BlendRLRenderer(BaseRenderer):
    model: Union[NsfrActorCritic, ActorCritic]
    window: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self,
                 agent_path = None,
                 env_name = "seaquest",
                 fps: int = 15,
                 device = "cpu",
                 screenshot_path = "",
                 render_panes=True,
                 seed=0,
                 deterministic=True,
                 env_kwargs: dict = None,):
        super().__init__(agent_path, env_name, fps, device, screenshot_path, render_panes, seed)

        self.deterministic = deterministic

        # Load model and environment
        self.model = load_model(agent_path, env_kwargs_override=env_kwargs, device=device)
        self.env = NudgeBaseEnv.from_name(env_name, mode='deictic', seed=self.seed, **env_kwargs)
        # self.env = self.model.env
        self.env.reset()
        print(self.model._print())

        print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            self.keys2actions = self.env.env.unwrapped.get_keys_to_action()
        except Exception:
            print(yellow("Info: No key-to-action mapping found for this env. No manual user control possible."))
            self.action_meanings = None
            self.keys2actions = {}

        self.current_frame = self._get_current_frame()
        self.predicates = self.model.logic_actor.prednames
        self._init_pygame(self.current_frame)

    def run(self):
        length = 0
        ret = 0

        obs, obs_nn = self.env.reset()
        obs_nn = torch.tensor(obs_nn, device=self.model.device)

        while self.running:
            self.reset = False
            self._handle_user_input()
            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.human_playing:  # human plays game manually
                    action = self._get_action()
                else:  # AI plays the game
                    action, logprob = self.model.act(obs_nn, obs)  # update the model's internals
                    value = self.model.get_value(obs_nn, obs)
                self.action = action

                (new_obs, new_obs_nn), reward, done, terminations, infos = self.env.step(action,
                                                                                         is_mapped=self.human_playing)
                # if reward > 0:
                # print(f"Reward: {reward:.2f}")
                new_obs_nn = torch.tensor(new_obs_nn, device=self.model.device)
                self.current_frame = self._get_current_frame()

                self._render()

                if self.human_playing and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs = self.env.reset()
                    self._render()

                obs = new_obs
                obs_nn = new_obs_nn
                length += 1

                if done:
                    print(f"Return: {ret} - Length {length}")
                    ret = 0
                    length = 0
                    self.env.reset()

        pygame.quit()

    def _get_current_frame(self):
        return self.env.env.render()

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the entire window
        self._render_policy_probs()
        self._render_predicate_probs()
        self._render_neural_probs()
        self._render_env()

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_policy_probs_rows(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * self.cell_background_highlight_policy + (1 - w_i) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                self.panes_col_width - 12,
                28
            ])

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

    def _render_policy_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * self.cell_background_highlight_policy + (1 - w_i) * self.cell_background_selected
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + i * 500,
                anchor[1] - 2,
                (self.panes_col_width / 2 - 12) * w_i,
                28
            ])

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            if i == 0:
                text_rect.topleft = (self.env_render_shape[0] + 10, 25)
            else:
                text_rect.topleft = (self.env_render_shape[0] + 10 + i * 500, 25)
            self.window.blit(text, text_rect)

    def _render_predicate_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        nsfr = self.model.actor.logic_actor
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        for i, (pred, val) in enumerate(pred_vals.items()):
            i += 2
            # Render cell background
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + self.panes_col_width / 2,
                anchor[1] - 2 + i * 35,
                (self.panes_col_width / 2 - 12) * val,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10 + self.panes_col_width / 2, 25 + i * 35)
            self.window.blit(text, text_rect)

    def _render_neural_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)
        blender_actor = self.model.actor
        action_vals = blender_actor.neural_action_probs[0].detach().cpu().numpy()
        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft",
                        "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire",
                        "downleftfire"]
        for i, (pred, val) in enumerate(zip(action_names, action_vals)):
            i += 2
            # Render cell background
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                (self.panes_col_width / 2 - 12) * val,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

    def _render_facts(self, th=0.1):
        anchor = (self.env_render_shape[0] + 10, 25)

        # nsfr = self.nsfr_reasoner
        nsfr = self.model.actor.logic_actor

        fact_vals = {}
        v_T = nsfr.V_T[0]
        preds_to_skip = ['.', 'true_predicate', 'test_predicate_global', 'test_predicate_object']
        for i, atom in enumerate(nsfr.atoms):
            if v_T[i] > th:
                if atom.pred.name not in preds_to_skip:
                    fact_vals[atom] = v_T[i].item()

        for i, (fact, val) in enumerate(fact_vals.items()):
            i += 2
            # Render cell background
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                self.panes_col_width - 12,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {fact}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)

