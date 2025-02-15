import time
import warnings

import pygame

from typing import Union

import torch

from framework_utils.BaseRenderer import BaseRenderer
from ns_policies.blendrl.nudge.agents.logic_agent import NsfrActorCritic
from ns_policies.blendrl.nudge.agents.neural_agent import ActorCritic
from ns_policies.blendrl.nudge.env import NudgeBaseEnv
from ns_policies.blendrl.nudge.utils import load_model, yellow, get_program_nsfr


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
                 print_rewards=False,
                 render_panes=True,
                 lst_panes=None,
                 seed=0,
                 deterministic=True,
                 env_kwargs: dict = None,):
        super().__init__(agent_path=agent_path,
                         env_name=env_name,
                         fps=fps, device=device,
                         screenshot_path=screenshot_path,
                         print_rewards=print_rewards,
                         render_panes=render_panes,
                         lst_panes=lst_panes,
                         seed=seed)

        if lst_panes is None:
            lst_panes = []
        self.deterministic = deterministic

        # Load model and environment
        self.model = load_model(agent_path, env_kwargs_override=env_kwargs, device=device)
        self.env = NudgeBaseEnv.from_name(env_name, mode='blendrl', seed=self.seed, **env_kwargs)
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

        self.overlay = env_kwargs["render_oc_overlay"]

    def run(self):

        obs, obs_nn = self.env.reset()
        obs = obs.to(self.device)
        obs_nn = torch.tensor(obs_nn, device=self.model.device)

        while self.running:
            self.reset = False
            self._handle_user_input()
            self.env.env.render_oc_overlay = self.overlay

            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.human_playing:  # human plays game manually
                    action = self._get_action()
                    time.sleep(0.05)
                else:  # AI plays the game
                    action, logprob = self.model.act(obs_nn, obs)  # update the model's internals
                    value = self.model.get_value(obs_nn, obs)
                self.action = action

                (new_obs, new_obs_nn), reward, done, terminations, infos = self.env.step(action,
                                                                                         is_mapped=self.human_playing)

                new_obs_nn = torch.tensor(new_obs_nn, device=self.model.device)
                self.current_frame = self._get_current_frame()

                self._render()

                if float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    new_obs = self.env.reset()
                    self._render()

                new_obs = new_obs.to(self.device)
                obs = new_obs
                obs_nn = new_obs_nn

        pygame.quit()

    def _get_current_frame(self):
        return self.env.env.render()

    def _render(self):
        """
        Render window
        """
        lst_possible_panes = ["policy", "selected_actions", "semantic_actions"]
        self.window.fill((0,0,0))  # clear the entire window
        self._render_env()

        anchor = (self.env_render_shape[0] + 10, 25)

        # render neural and logic policy
        if "policy" in self.lst_panes:
            pane_size1 = self._render_policy_probs(anchor)
            pane_size2 = self._render_neural_probs((anchor[0], anchor[1] + pane_size1[1]))
            pane_size3 = self._render_predicate_probs((anchor[0]+ pane_size2[0], anchor[1] + pane_size1[1]))
            anchor = (anchor[0], anchor[1] + 10 + pane_size1[1] + max(pane_size2[1], pane_size3[1]))

        # render selected actions
        if "selected_actions" in self.lst_panes:
            pane_size = self._render_selected_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1]+pane_size[1])
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])

        # render semantic actions
        if "semantic_actions" in self.lst_panes:
            pane_size = self._render_semantic_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1]+pane_size[1])
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])

        # TODO add logic rule evaluation

        remains = [pane for pane in self.lst_panes if pane not in lst_possible_panes]
        if remains:
            warnings.warn(f"No panes available for {remains} in blendRL! Possible panes are: {lst_possible_panes}", UserWarning)

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_policy_probs(self, anchor):
        row_height = self._get_row_height()

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * self.cell_background_highlight_policy + (1 - w_i) * self.cell_background_selected
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + i * self.panes_col_width / 2,
                anchor[1] - 2,
                (self.panes_col_width / 2 - 12) * w_i,
                self.font.get_height() + 4
            ])

            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            if i == 0:
                text_rect.topleft = anchor
            else:
                text_rect.topleft = (anchor[0] + self.panes_col_width / 2, anchor[1])
            self.window.blit(text, text_rect)
        return (self.panes_col_width, row_height)

    def _render_predicate_probs(self, anchor):
        row_height = self._get_row_height()
        row_cnter = 0

        nsfr = self.model.actor.logic_actor
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        for i, (pred, val) in enumerate(pred_vals.items()):
            # Render cell background
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 2 - 12) * val,
                self.font.get_height() + 4
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)
            self.window.blit(text, text_rect)
            row_cnter += 1
        return (self.panes_col_width / 2, row_height * row_cnter)

    def _render_neural_probs(self, anchor):
        row_height = self.font.get_height()
        row_height += row_height/2
        row_cnter = 0

        blender_actor = self.model.actor
        action_vals = blender_actor.neural_action_probs[0].detach().cpu().numpy()
        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft",
                        "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire",
                        "downleftfire"]
        for i, (pred, val) in enumerate(zip(action_names, action_vals)):
            # i += 2
            # Render cell background
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 2 - 12) * val,
                self.font.get_height() + 4
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)
            self.window.blit(text, text_rect)
            row_cnter += 1
        return (self.panes_col_width / 2, row_height * row_cnter)

    def _render_logic_rules(self, anchor):
        """
        Render logic action rules and highlight the selected rule.
        """
        row_height = self.font.get_height()
        row_height += row_height/2
        row_cnter = 0

        logic_action_rules = get_program_nsfr(self.model.logic_actor)

        title = self.font.render("Logic Action Rules", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = anchor
        self.window.blit(title, title_rect)
        row_cnter += 1

        anchor = (anchor[0], anchor[1] + row_height)

        predicate_indices = []
        action_logic_prob = 0

        action = self.action_meanings[self.action].lower()
        basic_actions = self.model.actor.env.pred2action.keys()
        action_indices = self.model.actor.env.pred2action
        action_predicates = self.model.actor.env_action_id_to_action_pred_indices  # Dictionary of actions and its predicates.

        if action in basic_actions:
            # If selected action is a basic action, then it has predicates that contributed to its probability distribution.
            predicate_indices = action_predicates[action_indices[action]]
            action_logic_prob = self.model.actor.logic_action_probs[0].tolist()[
                action_indices[action]]  # Probability of the selected action given logic probability distribution.

            # Partly taken from to_action_distribution() in blender_agent.py.
            indices = torch.tensor(action_predicates[action_indices[action]])  # Indices of the predicates of selected action.
            indices = indices.expand(self.model.actor.batch_size, -1)
            indices = indices.to(self.model.actor.device)
            gathered = torch.gather(torch.logit(self.model.actor.raw_action_probs, eps=0.01), 1, indices)

            # Normalized probabilities of the predicates of selected action that they have assigned to it.
            predicate_probs = torch.softmax(gathered, dim=1).cpu().detach().numpy()[ 0]
            pred2prob_dict = {}  # Key is index of the predicate and value is the probability that it has assigned to the action.
            for j in range(len(indices.tolist()[0])):
                pred2prob_dict[indices.tolist()[0][j]] = predicate_probs[j]

        # Determines how much influence the logic action probabilities have on the overall action probability distribution.
        logic_policy_weight = self.model.actor.w_policy[1].tolist()

        for i, rule in enumerate(logic_action_rules):
            is_selected = 0
            if i in predicate_indices and self.model.actor.actor_mode != "neural" and action_logic_prob * logic_policy_weight > 0.1 and \
                    pred2prob_dict[i] > 0.1:
                # Highlight predicates that contributed to the probability of the selected action with their assignment of a probability bigger than 0.1.
                # Another condition is a large enough weight for the logic policy during its blending with the neural module, as well as logic probability of the action.
                is_selected = 1

            color = is_selected * self.cell_background_highlight + (1 - is_selected) * self.cell_background_default
            # Render cell background
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 2 - 12) * is_selected,
                self.font.get_height() + 4
            ])

            text = self.font.render(rule, True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)
            self.window.blit(text, text_rect)
            row_cnter += 1
        return (self.panes_col_width / 2, row_height * row_cnter)  # width, height

