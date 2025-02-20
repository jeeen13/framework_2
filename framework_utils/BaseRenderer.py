import numpy as np
import pygame

from datetime import datetime
from abc import abstractmethod, ABC

try:
    from pygame_screen_record import ScreenRecorder
    _screen_recorder_imported = True
except ImportError as imp_err:
    _screen_recorder_imported = False


class BaseRenderer(ABC):
    """
    Base class for rendering an agent interacting with an environment using Pygame.

    This class manages rendering, user interaction, and simulation control. It provides
    methods for handling keyboard inputs, rendering overlays, and displaying panes
    with information about the agent's actions. The rendering can be adjusted with
    zoom settings, and additional information panes can be toggled.
    """
    window: pygame.Surface
    clock: pygame.time.Clock
    zoom: int = 5

    def __init__(self,
                 agent_path = None,
                 env_name = "",
                 fps: int = 15,
                 device = "cpu",
                 screenshot_path = "",
                 print_rewards=False,
                 render_panes=True,
                 lst_panes=None,
                 seed = 0):
        """
        Initialize the BaseRenderer with the rendering configuration

        :param agent_path: path to the trained agent
        :param env_name: Name of the environment to render
        :param fps: frames per second for rendering
        :param device: The computation device to use ("cpu" or "gpu", default: "cpu")
        :param screenshot_path: Directory where screenshots will be saved.
        :param print_rewards: Whether to print rewards to the console
        :param render_panes: Whether to render additional information panes (default: True).
        :param lst_panes: List of panes to display in the UI (default: None)
        :param seed: Random seed for environment initialization (default: 0).
        """

        self.fps = fps
        self.seed = seed
        self.render_panes = render_panes
        self.lst_panes = [] if lst_panes is None else lst_panes
        self.print_rewards = print_rewards
        self.panes_col_width = 500 * 2
        self.cell_background_default = np.array([40, 40, 40])
        self.cell_background_selected = np.array([80, 80, 80])
        self.cell_background_highlight =  np.array([40, 150, 255])
        self.cell_background_highlight_policy = np.array([234, 145, 152])

        self.current_keys_down = set()
        self.current_mouse_pos = None

        self.agent_path = agent_path
        self.env_name = env_name
        self.model = None
        self.env = None
        self.device = device
        self.screenshot_path = screenshot_path

        self.action_meanings = None
        self.keys2actions = {}
        self.action = None
        self.current_frame = None

        self.active_cell_idx = None
        self.candidate_cell_ids = []
        self.current_active_cell_input : str = ""

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.human_playing = False
        self.overlay = True
        self.reset = False
        self._recording = False

    def _init_pygame(self, sample_image):
        """
        Initialize the Pygame window and rendering settings.

        :param sample_image: A sample frame from the environment to determine the rendering dimensions.
        """

        pygame.init()
        pygame.display.set_caption("Environment")

        # Determine the dimension of the window
        if sample_image.shape[0] > sample_image.shape[1]:
            sample_image = np.repeat(np.repeat(np.swapaxes(sample_image, 0, 1), self.zoom, axis=0), self.zoom, axis=1)
        self.env_render_shape = sample_image.shape[:2]
        window_size = list(self.env_render_shape[:2])

        # Extend the window for additional panes
        if self.render_panes:
            window_size[0] += self.panes_col_width

        self.window = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 16)

    @abstractmethod
    def run(self):
        """
        Runs the agent in the environment while handling user input and rendering.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def _get_current_frame(self):
        """
        Return the current frame from the environment
        :return: frame
        """
        raise NotImplementedError

    def _get_action(self):
        """
        Determines the action to be taken based on the current user input.

        :return: action corresponding to the pressed keys
        """
        if self.keys2actions is None:
            return 0  # NOOP
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        """
        Handles user input events from the keyboard and mouse.
        """
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.env.reset()

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = not self.fast_forward

                elif event.key == pygame.K_h:  # 'H': toggle human playing
                    self.human_playing = not self.human_playing
                    if self.human_playing:
                        print("Human playing")
                    else:
                        print("AI playing")

                elif event.key == pygame.K_o:  # 'O': toggle overlay
                    self.overlay = not self.overlay

                elif event.key == pygame.K_m:  # 'M': save snapshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    pygame.image.save(self.window, self.screenshot_path + file_name)

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

                elif pygame.K_0 <= event.key <= pygame.K_9:  # enter digit
                    char = str(event.key - pygame.K_0)
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input += char

                elif event.key == pygame.K_BACKSPACE:  # remove character
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input = self.current_active_cell_input[:-1]

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

    @abstractmethod
    def _render(self):
        """
        Renders the environment, overlays and panes onto the Pygame window.
        """
        raise NotImplementedError

    def _render_env(self):
        """
        Render environment frame onto the Pygame window.
        """
        frame = self.current_frame
        if frame.shape[0] > frame.shape[1]:
            frame = np.swapaxes(np.repeat(np.repeat(frame, self.zoom, axis=0), self.zoom, axis=1), 0, 1)
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))

    def _render_selected_action(self, anchor):
        """
        Render selected action pane at anchor point in window

        :param anchor: The (x, y) coordinates of the top-left corner in the window where the action pane should be rendered.
        :return: (width, height) of the rendered pane
        """
        row_height = self._get_row_height()
        row_cnter = 0
        action_names = ["noop", "fire", "up", "right", "left", "down", "upright", "upleft", "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire", "uprightfire", "upleftfire", "downrightfire", "downleftfire"]

        # Render title
        title = self.font.render("Raw Selected Action", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = anchor
        self.window.blit(title, title_rect)
        row_cnter += 1

        anchor = (anchor[0], anchor[1] + row_height)

        # Render actions
        for i, action in enumerate(action_names):
            is_selected = 0
            if action.upper() == self.action_meanings[self.action[0]]:
                is_selected = 1 # Only the selected action will be highlighted.

            color = is_selected * self.cell_background_highlight + (1 - is_selected) * self.cell_background_default
            # Render cell background
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 4  - 12) * is_selected,
                self.font.get_height() + 4
            ])

            text = self.font.render(action, True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)
            self.window.blit(text, text_rect)
            row_cnter += 1

        return self.panes_col_width / 4, row_height * row_cnter  # width, height

    def _render_semantic_action(self, anchor):
        """
        Render semantic action pane at anchor point in window

        :param anchor: The (x, y) coordinates of the top-left corner in the window where the action pane should be rendered.

        :return: (width, height) of the rendered pane
        """
        row_height = self._get_row_height()
        row_cnter = 0

        # Render title
        title = self.font.render("Semantic Actions", True, "white", None)
        title_rect = title.get_rect()
        title_rect.topleft = anchor
        self.window.blit(title, title_rect)
        row_cnter += 1

        anchor = (anchor[0], anchor[1] + row_height)

        # Render actions
        action_names = ["noop", "fire", "up", "right", "left", "down"]
        action = self.action_meanings[self.action[0]].lower()
        for i, action_name in enumerate(action_names):
            include = 1 if action_name in action else 0
            color = include * self.cell_background_highlight + (1 - include) * self.cell_background_default
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * row_height,
                (self.panes_col_width / 4 - 12) * include,
                self.font.get_height() + 4
            ])

            text = self.font.render(action_name, True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (anchor[0], anchor[1] + i * row_height)  # Place it at the bottom.
            self.window.blit(text, text_rect)
            row_cnter += 1

        return self.panes_col_width / 4, row_height * row_cnter  # width, height

    def _get_row_height(self):
        """
        Computes and returns the height of a row used for rendering text-based UI elements.

        :return: computed row height
        """
        row_height = self.font.get_height()
        row_height += row_height / 2
        return row_height
