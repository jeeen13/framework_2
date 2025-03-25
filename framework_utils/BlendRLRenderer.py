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

import copy

import ns_policies.blendrl.saliency as saliency
import numpy as np


class BlendRLRenderer(BaseRenderer):
    """
    Renderer class for BlendRL agents, handling visualization, recording, and interaction.
    """
    model: Union[NsfrActorCritic, ActorCritic]

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
                 deterministic=False,
                 env_kwargs: dict = None,):
        """
        :param agent_path: path to the trained agent
        :param env_name: name of the environment
        :param fps: frames per second
        :param device: The computation device to use ("cpu" or "gpu", default: "cpu")
        :param screenshot_path: Directory where screenshots will be saved.
        :param print_rewards: Whether to print rewards to the console.
        :param render_panes: Whether to render additional information panes (default: True).
        :param lst_panes: List of panes to display in the UI (default: None).
        :param seed: Random seed for environment initialization (default: 0).
        :param deterministic: Whether to use deterministic policy (default: False).
        :param env_kwargs: environment keyword arguments (default: None).
        """
        super().__init__(agent_path=agent_path,
                         env_name=env_name,
                         fps=fps, device=device,
                         screenshot_path=screenshot_path,
                         print_rewards=print_rewards,
                         render_panes=render_panes,
                         lst_panes=lst_panes,
                         seed=seed)

        ################################################################################
        # LOAD MODEL AND ENVIRONMENT
        self.model = load_model(agent_path, env_kwargs_override=env_kwargs, device=device)
        self.env = NudgeBaseEnv.from_name(env_name, mode='blendrl', seed=self.seed, **env_kwargs)
        self.env.reset()
        print(self.model._print())
        self.deterministic = deterministic

        print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

        #################################################################################
        # RENDERER INITIALIZATION
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

        self.blending_weights = []

        self.history = {'ins': [], 'obs': []} # For heat map

    def run(self):

        obs, obs_nn = self.env.reset()
        obs = obs.to(self.device)
        obs_nn = torch.tensor(obs_nn, device=self.model.device)

        self.heat_counter = -1

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
                    action, logprob, blending_weights_new = self.model.act(obs_nn, obs)  # update the model's internals
                    #action_probs, blending_weights_new = self.model.actor(obs_nn, obs)
                    value = self.model.get_value(obs_nn, obs)
                    self.blending_weights = blending_weights_new.to(self.device)
                self.action = action

                (new_obs, new_obs_nn), reward, done, terminations, infos = self.env.step(action,
                                                                                         is_mapped=self.human_playing)

                new_obs_nn = torch.tensor(new_obs_nn, device=self.model.device)
                self.current_frame = self._get_current_frame()

                if "heat_map" in self.lst_panes:
                    self.heat_counter += 1
                    self.update_history()

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
        lst_possible_panes = ["policy", "selected_actions", "semantic_actions", "logic_action_rules", "logic_valuations", "state_usage", "heat_map"]
        self.window.fill((0,0,0))  # clear the entire window
        if "heat_map" in self.lst_panes:
            self._render_heat_map()
        else:
            self._render_env()
        anchor = (self.env_render_shape[0] + 10, 25)

        # render neural and logic policy
        if "policy" in self.lst_panes:
            pane_size1 = self._render_policy_probs(anchor)
            pane_size2 = self._render_neural_probs((anchor[0], anchor[1] + pane_size1[1]))
            pane_size3 = self._render_predicate_probs((anchor[0]+ pane_size2[0], anchor[1] + pane_size1[1]))
            anchor = (anchor[0], anchor[1] + 10 + pane_size1[1] + max(pane_size2[1], pane_size3[1]))

        panes_row = []
        # render selected actions
        if "selected_actions" in self.lst_panes:
            pane_size = self._render_selected_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + pane_size[1] + 10)
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        # render semantic actions
        if "semantic_actions" in self.lst_panes:
            pane_size = self._render_semantic_action(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        if "logic_valuations" in self.lst_panes:
            pane_size = self._render_logic_valuations(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        if "logic_action_rules" in self.lst_panes:
            pane_size = self._render_logic_rules(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        # render state usage
        if "state_usage" in self.lst_panes:
            pane_size = self.render_state_usage(anchor)
            if anchor[0] + pane_size[0] >= self.window.get_width():
                anchor = (self.env_render_shape[0] + 10, anchor[1] + max(panes_row) + 10)
                panes_row = []
            else:
                anchor = (anchor[0] + pane_size[0], anchor[1])
                panes_row.append(pane_size[1])

        remains = [pane for pane in self.lst_panes if pane not in lst_possible_panes]
        if remains:
            warnings.warn(f"No panes available for {remains} in blendRL! Possible panes are: {lst_possible_panes}", UserWarning)

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_policy_probs(self, anchor):
        """
        Render the policy probabilities as colored bars with text labels.

        :param anchor: The (x, y) coordinates of the top-left corner in the window where the action pane should be rendered.
        :return: The width and height of the rendered pane
        """
        row_height = self._get_row_height()

        model = self.model
        policy_names = ['neural', 'logic']
        weights = model.get_policy_weights()
        for i, w_i in enumerate(weights):
            w_i = w_i.item()
            name = policy_names[i]
            # Render cell background
            color = w_i * self.cell_background_highlight_policy + (1 - w_i) * self.cell_background_selected
            # Draw rectangle
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2 + i * self.panes_col_width / 2,
                anchor[1] - 2,
                (self.panes_col_width / 2 - 12) * w_i,
                self.font.get_height() + 4
            ])

            # Render policy type
            text = self.font.render(str(f"{w_i:.3f} - {name}"), True, "white", None)
            text_rect = text.get_rect()
            if i == 0:
                text_rect.topleft = anchor
            else:
                text_rect.topleft = (anchor[0] + self.panes_col_width / 2, anchor[1])
            self.window.blit(text, text_rect)
        return (self.panes_col_width, row_height)

    def _render_predicate_probs(self, anchor):
        """
        Render the predicate probabilities as colored bars with text labels.

        :param anchor: The (x, y) coordinates of the top-left corner in the window where the action pane should be rendered.
        :return: The width and height of the rendered pane
        """
        row_height = self._get_row_height()
        row_cnter = 0

        nsfr = self.model.actor.logic_actor
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        for i, (pred, val) in enumerate(pred_vals.items()):
            # Determine background color based on predicate probability
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            # Draw rectangle
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 2 - 12) * val,
                self.font.get_height() + 4
            ])

            # Render predicate probability text
            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)
            self.window.blit(text, text_rect)
            row_cnter += 1
        return (self.panes_col_width / 2, row_height * row_cnter)

    def _render_neural_probs(self, anchor):
        """
        Render neural action probabilities as colored bars with text labels.

        :param anchor: The (x, y) coordinates of the top-left corner in the window where the action pane should be rendered.
        :return: The width and height of the rendered pane
        """
        row_height = self.font.get_height()
        row_height += row_height/2
        row_cnter = 0

        blender_actor = self.model.actor
        action_vals = blender_actor.neural_action_probs[0].detach().cpu().numpy()
        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft",
                        "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire",
                        "downleftfire"]
        for i, (pred, val) in enumerate(zip(action_names, action_vals)):
            # Render cell background
            color = val * self.cell_background_highlight + (1 - val) * self.cell_background_default
            # Draw rectangle
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 2 - 12) * val,
                self.font.get_height() + 4
            ])

            # Render actions
            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)
            self.window.blit(text, text_rect)
            row_cnter += 1
        return (self.panes_col_width / 2, row_height * row_cnter)

    def selected_logic_rule(self):
        predicate_indices = []
        action_logic_prob = 0
        pred2prob_dict = {}  # Key is index of the predicate and value is the probability that it has assigned to the action.

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
            indices = torch.tensor(
                action_predicates[action_indices[action]])  # Indices of the predicates of selected action.
            indices = indices.expand(self.model.actor.batch_size, -1)
            indices = indices.to(self.model.actor.device)
            gathered = torch.gather(torch.logit(self.model.actor.raw_action_probs, eps=0.01), 1, indices)

            predicate_probs = torch.softmax(gathered, dim=1).cpu().detach().numpy()[0]  # Normalized probabilities of the predicates of selected action that they have assigned to it.
            for j in range(len(indices.tolist()[0])):
                pred2prob_dict[indices.tolist()[0][j]] = predicate_probs[j]

        return predicate_indices, action_logic_prob, pred2prob_dict

    def _render_logic_rules(self, anchor):
        """
        Render logic action rules and highlight the selected rule.
        """
        row_height = self.font.get_height()
        row_height += row_height/2
        row_center = 0

        logic_action_rules = get_program_nsfr(self.model.logic_actor)

        title = self.font.render("Logic Action Rules", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = anchor
        self.window.blit(title, title_rect)
        row_center += 1

        anchor = (anchor[0], anchor[1] + row_height)

        predicate_indices, action_logic_prob, pred2prob_dict = self.selected_logic_rule()

        # Determines how much influence the logic action probabilities have on the overall action probability distribution.
        logic_policy_weight = self.model.actor.w_policy[1].tolist()

        for i, rule in enumerate(logic_action_rules):
            is_selected = 0
            if i in predicate_indices and self.model.actor.actor_mode != "neural" and action_logic_prob * logic_policy_weight > 0.1 and \
                    pred2prob_dict[i] > 0.1:
                # Highlight predicates that contributed to the probability of the selected action with their assignment of a normalized probability bigger than 0.1.
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
            row_center += 1
        return (self.panes_col_width / 2, row_height * row_center)  # width, height

    def all_object_pairs(self, all_rule_state_atoms, grounded_rule, current_obj, predicates_obj, predicates):
        '''
        Recursive function for determining all grounded rules.
        '''

        if all_rule_state_atoms == []:  # Base case. All atoms of the grounded rule have been iterated. Append the grounded rule [(right_of_diver(obj1,obj2), 0.7), (visible_diver(obj2), 0.9), (not_full_divers(img), 1)] for example to the list of all rules.
            self.grounded_rule_atoms.append(grounded_rule)

        else:
            fixed_obj = 0

            new_atom_obj = predicates_obj[predicates.index(all_rule_state_atoms[0][0][0].pred.name)].terms  # List of objects of the new atom. If it is right_of_diver, then the list is [P, D].

            obj_indices = []  # Indices of objects that are already set by previous atoms. If a previous atom has set {P: obj1} in current_obj, then this list contains [0] if P is the first key in the dictionary.

            for obj in new_atom_obj:
                if obj in current_obj:
                    fixed_obj += 1
                    obj_indices.append(list(current_obj.keys()).index(obj))

            if fixed_obj == len(
                    new_atom_obj):  # All objects of the state atom are already set by previous atoms and have to be the same. There can only be one grounded atom that fulfills this condition in the list. For example, atom is right_of_diver(P, D) and the objects are already in current_obj.
                for atom in all_rule_state_atoms[0]:  # Iterate over atom list. [(right_of_diver(obj1,obj2), 0.7), (right_of_diver(obj1,obj3), 0.8), (right_of_diver(obj1,obj4), 0.2), (right_of_diver(obj1,obj5), 0.1)]
                    if all(atom_obj in current_obj.values() for atom_obj in atom[0].terms):  # If all objects of atom match values of the current_obj dictionary, then atom is appended to grounded rule.
                        grounded_rule.append(atom)
                        self.all_object_pairs(all_rule_state_atoms[1:], grounded_rule, current_obj, predicates_obj, predicates)  # Recursion, go over next atoms with list that contains newly added atom. E.g. all_rule_state_atoms now is [[(visible_diver(obj2), 0.9), (visible_diver(obj3), 0.3) (visible_diver(obj4), 0.1)], [(not_full_divers(img), 1)]]

            elif fixed_obj < len(
                    new_atom_obj):  # At least one object of the state atom has not been set by previous atoms. Multiple grounded atoms can fulfill this condition, therefore every combination has to be checked. E.g. P and D are set, but atom has object X.
                for atom in all_rule_state_atoms[0]:
                    # Go through all grounded atoms of the current atom and see which object combinations match.
                    matching_obj = 0
                    new_obj_indices = []
                    for i, atom_obj in enumerate(atom[0].terms):
                        # Check if atom has correct number of matching objects to current_obj as it's supposed to. E.g. if P and D have been set, and we have an atom that has objects P and X, then one should match.
                        if atom_obj in current_obj.values():
                            matching_obj += 1
                        else:
                            new_obj_indices.append(i)
                    if matching_obj == fixed_obj:
                        new_current_obj = copy.deepcopy(current_obj)
                        new_grounded_rule = copy.deepcopy(grounded_rule)
                        for index in new_obj_indices:
                            new_current_obj[new_atom_obj[index]] = atom[0].terms[index]  # Add new objects to current_obj dictionary. {P: obj1, D: obj3, X: obj5}

                        new_grounded_rule.append(atom)  # Update grounded rule list with new atom. E.g. [(right_of_diver(obj1,obj2), 0.7), (visible_diver(obj2), 0.9)]

                        self.all_object_pairs(all_rule_state_atoms[1:], new_grounded_rule, new_current_obj, predicates_obj, predicates)

    def _render_logic_valuations(self, anchor):
        '''
        Render logic state valuations by showcasing all logic action rules and their state atoms, but only showing the truth values of the selected rules.
        '''
        row_height = self.font.get_height()
        row_height += row_height/2
        row_center = 0

        title = self.font.render("Logic State Valuations", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = anchor
        self.window.blit(title, title_rect)
        row_center += 1

        anchor = (anchor[0], anchor[1] + row_height)

        # Get the current logic valuations
        valuation = self.model.logic_actor.V_T
        batch = valuation[0].detach().cpu().numpy()  # List of values of all state atoms.

        rules_and_predicates = {}  # Contains logic action rules and their corresponding state atoms. {up_ladder(X): [on_ladder(P,L), same_level_ladder(P,L)], ...}

        grounded_rules = {}  # Contains each logic action rule and their highest grounded rule. E.g. {up_ladder: [(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)], left_ladder: [(right_of_ladder(x1,x2), 0.2), (same_level_ladder(x1,x2), 0.9)], right_ladder: [(left_of_ladder(x1,x2), 0.9), (same_level_ladder(x1,x2), 0.9)]}

        for clause in self.model.logic_actor.clauses:
            # Go through every logic action rule.
            predicates_obj = []
            predicates = []
            for predicate in clause.body:
                # Collect the state atoms of the logic action rule in a list.
                predicates_obj.append(predicate)
                predicates.append(predicate.pred.name)

            rules_and_predicates[clause.head] = predicates_obj

            objects = set()  # Set of objects of the clause. {P,L}
            for predicate in rules_and_predicates[clause.head]:
                objects.update(predicate.terms)

            all_rule_state_atoms = []  # Contains all state atoms of the rule with all object combinations. List of lists. [[(on_ladder(x1,x2), 0.8), (on_ladder(x1,x3), 0.4)], [(same_level_ladder(x1,x2), 0.9), (same_level_ladder(x1,x3), 0.2)]] for rule up_ladder.

            for predicate in predicates_obj:
                rule_atom = []  # List of a state atom of the current rule for all object combinations. [(on_ladder(x1,x2), 0.8), (on_ladder(x1,x3), 0.4)]
                for i, atom_value in enumerate(batch):
                    if self.model.logic_actor.atoms[i].pred.name == predicate.pred.name:
                        rule_atom.append((self.model.logic_actor.atoms[i],
                                          atom_value))  # Tuple of atom and its value. (on_ladder(x1,x2), 0.8)
                all_rule_state_atoms.append(rule_atom)

            all_rule_state_atoms.sort(key=lambda atom: len(atom[0][0].terms), reverse=True)

            self.grounded_rule_atoms = []  # Contains the atoms of all grounded rules. [[(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)], [(on_ladder(x1,x3), 0.4), (same_level_ladder(x1,x3), 0.2)]]
            grounded_rule = []  # Serves as a list for the grounded rule in the combination algorithm, which will be appended to grounded_rule_atoms. E.g. [(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)].
            current_obj = {}  # Contains objects that have already been set in the combination algorithm. E.g. {P: obj1, D: obj2}
            self.all_object_pairs(all_rule_state_atoms, grounded_rule, current_obj, predicates_obj, predicates)

            product_grounded_rules = []  # Collects the product of the state atoms of all grounded rules. [0.8*0.9, 0.4*0.2]

            for grounded_rule in self.grounded_rule_atoms:
                product_atoms = 1
                for atom in grounded_rule:
                    product_atoms *= atom[1]
                product_grounded_rules.append(product_atoms)

            grounded_rules[clause.head] = self.grounded_rule_atoms[product_grounded_rules.index(max(product_grounded_rules))]  # Add grounded rule that has highest value. E.g. {up_ladder: [(on_ladder(x1,x2), 0.8), (same_level_ladder(x1,x2), 0.9)]}

        index = -1.5

        predicate_indices, action_logic_prob, pred2prob_dict = self.selected_logic_rule()

        logic_policy_weight = self.model.actor.w_policy[1].tolist()  # Determines how much influence the logic action probabilities have on the overall action probability distribution.

        rule_index = 0
        for rule, atoms in grounded_rules.items():

            index += 1.5
            rule_title = f"Clause: {rule.pred.name}"
            title = self.font.render(rule_title, True, "white", None)
            title_rect = title.get_rect()
            title_rect.topleft = (anchor[0], anchor[1] + index * row_height)
            self.window.blit(title, title_rect)
            row_center += 1

            if rule_index in predicate_indices and self.model.actor.blender_mode != "neural" and action_logic_prob * logic_policy_weight > 0.1 and \
                    pred2prob_dict[rule_index] > 0.1:
                # Only highlight truth values of atoms from logic action rules that participated in the selection of the action with their assignment of a normalized probability bigger than 0.1.
                # Another condition is a large enough weight for the logic policy during its blending with the neural module, as well as logic probability of the action.
                for atom in atoms:
                    index += 1
                    # Render cell background
                    color = atom[1] * self.cell_background_highlight + (1 - atom[1]) * self.cell_background_default
                    pygame.draw.rect(self.window, color, [
                        anchor[0] - 2,
                        anchor[1] - 2 + index * row_height,
                        (self.panes_col_width / 2 - 12) * atom[1],
                        self.font.get_height() + 4
                    ])

                    text = self.font.render(str(f"{atom[1]:.3f} - {atom[0].pred.name}"), True, "white", None)
                    text_rect = text.get_rect()
                    text_rect.topleft = (anchor[0], anchor[1] + index * row_height)
                    self.window.blit(text, text_rect)

            else:
                for atom in atoms:
                    index += 1
                    # Render cell background
                    text = self.font.render(str(f"{atom[0].pred.name}"), True, "white", None)
                    text_rect = text.get_rect()
                    text_rect.topleft = (anchor[0], anchor[1] + index * row_height)
                    self.window.blit(text, text_rect)

            row_center += 1
            rule_index += 1

        return (self.panes_col_width / 2, row_height * row_center)  # width, height

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
        rgb_obs = self.env.env._env.render() # Original rgb observation with shape (210,160,3)
        self.history['ins'].append(rgb_obs)
        self.history['obs'].append(self.env.env._env.observation(rgb_obs)) # shape (4,84,84), no prepro necessary