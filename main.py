import subprocess
import os
import shlex

def split_command(command: str):
    return shlex.split(command)

def run_command(raw_command, path):
    try:
        script_path = os.path.abspath(path)
        if not os.path.exists(script_path):
            print(f"Die Datei wurde nicht gefunden: {script_path}")
            return

        command = split_command(raw_command)
        command[1] = script_path

        subprocess.run(command, check=True)
        print("Befehl erfolgreich ausgeführt.")

    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen des Befehls: {e}")
    except FileNotFoundError as e:
        print(f"Fehler: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

def use_scobots():
    original_command = "python render_agent.py -g Pong -s 0 -r human -p default"
    path = "./ns_policies/SCoBOts_framework/render_agent.py"
    run_command(original_command, path)

def use_blendrl():
    original_command = "python3 play_gui.py --env-name donkeykong --agent-path ./ns_policies/blendrl/models/donkeykong_demo"
    path = "./ns_policies/blendrl/play_gui.py"
    run_command(original_command, path)

def main(policy_type):
    match policy_type:
        case "scobots":
            print("using scobots")
            use_scobots()
        case "blendrl":
            print("using blendrl")
            use_blendrl()
        case _:
            print("unknown policy type")


if __name__ == "__main__":
    #main("scobots")
    main("blendrl")
