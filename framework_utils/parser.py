import argparse

from distutils.util import strtobool

def render_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, required=True,
                        help="agent to use (e.g. 'scobots')")
    parser.add_argument("-ap", "--agent_path", type=str, required=True,
                        help="path to agent")
    parser.add_argument("-pl", "--panes_list", type=str, nargs="+",
                        help="list of panes (e.g. -pl policy semantic_actions selected_actions)")
    parser.add_argument("-f", "--fps", type=int, default=50)
    #########
    parser.add_argument("-e", "--environment", type=str, required=False, default="ocatari", choices=["ocatari", "hackatari"],
                        help="environment to load game (e.g. 'ocatari')")
    parser.add_argument("-g", "--game", type=str, required=True,
                        help="game to train (e.g. 'Pong')")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="seed")
    parser.add_argument("-r", "--reward", type=str, required=False, default="human", choices=["env", "human", "mixed"],
                        help="reward mode, env if omitted")
    # Scobots specific flags
    parser.add_argument("-p", "--prune", type=str, required=False, default="default", choices=["default", "external"],
                        help="use pruned focusfile (from default 'focusfiles' dir or external 'resources/focusfiles' dir. for custom pruning and or docker mount)")
    parser.add_argument("-x", "--exclude_properties",  action="store_true", help="exclude properties from feature vector")
    parser.add_argument("-n", "--version", type=str, required=False, help="specify which trained version. standard selects highest number")
    parser.add_argument("--rgb", required= False, action="store_true", help="rgb observation space")
    parser.add_argument("--record", required= False, action="store_true", help="wheter to record the rendered video")
    parser.add_argument("--nb_frames", type=int, default=0, help="stop recording after nb_frames (or 1 episode if not specified)")
    parser.add_argument("--print-reward", action="store_true", help="display the reward in the console (if not 0)")
    parser.add_argument("--viper", nargs="?", const=True, default=False, help="evaluate the extracted viper tree instead of a checkpoint")
    parser.add_argument("--hud", action="store_true", help="use HUD objects")
    # Neural Agent specific flags
    parser.add_argument("-fs", "--frameskip", type=int, default=4, help="Frames skipped after each action")
    parser.add_argument("-dp", "--dopamine_pooling", action='store_true', help="Use dopamine-like frameskipping")
    # Insight specific flags
    parser.add_argument("-rf", "--reward_function", type=str,  default="", help="Custom reward function file name")
    parser.add_argument("--resolution", type=int, default=84,  help="resolution")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gray", type=lambda x: bool(strtobool(x)), default=True, help="use gray or not")
    # HackAtari specific flags
    parser.add_argument("-m", "--modifs", nargs="+", default=[], help="List of modifications to the game")
    opts = parser.parse_args()


    env_str = "ALE/" + opts.game.capitalize() +"-v5"
    settings_str = ""
    pruned_ff_name = None
    hide_properties = False
    variant = "scobots"

    if opts.reward == "env":
        settings_str += "_reward-env"
    if opts.reward == "human":
        settings_str += "_reward-human"
    if opts.reward == "mixed":
        settings_str += "_reward-mixed"

    game_id = env_str.split("/")[-1].lower().split("-")[0]


    if opts.rgb:
        settings_str += "_rgb"
        variant= "rgb"
    else:
        settings_str += "_oc"

    if opts.prune:
        pruned_ff_name = f"pruned_{game_id}.yaml"
        variant =  "iscobots"
    if opts.prune == "default":
        settings_str += "_pruned"
    if opts.prune == "external":
        settings_str += "_pruned-external"
    if opts.exclude_properties:
        settings_str += '_excludeproperties'
        hide_properties = True

    exp_name = ""
    if opts.agent.lower() == "scobots":
        exp_name = opts.game + "_seed" + str(opts.seed) + settings_str

    return {
        "exp_name": exp_name,
        "env_str": env_str,
        "hide_properties": hide_properties,
        "pruned_ff_name": pruned_ff_name,
        "variant": variant,
        "version": opts.version or -1,
        "game": opts.game,
        "environment": opts.environment,
        "seed": opts.seed,
        "reward": opts.reward,
        "rgb": opts.rgb,
        "record": opts.record,
        "nb_frames": opts.nb_frames,
        "print_reward": opts.print_reward,
        "viper": opts.viper,
        "hud": opts.hud,
        "agent": opts.agent,
        "agent_path": opts.agent_path,
        "lst_panes": opts.panes_list,
        "fps": opts.fps,
        "frameskip": opts.frameskip,
        "dopamine_pooling": opts.dopamine_pooling,
        "threshold": opts.threshold,
        "gray": opts.gray,
        "resolution": opts.resolution,
        "modifs": opts.modifs,
        "reward_function": opts.reward_function,
    }