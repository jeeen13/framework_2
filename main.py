import subprocess
import os
import sys


def main():
    try:
        # Absoluter Pfad zur render_agent.py
        script_path = os.path.abspath("./ns_policies/SCoBOts_framework/render_agent.py")
        if not os.path.exists(script_path):
            print(f"Die Datei wurde nicht gefunden: {script_path}")
            return

        command = ["python", script_path, "-g", "Pong", "-s", "0", "-r", "human", "-p", "default"]

        # Subprozess starten
        subprocess.run(command, check=True)
        print("Befehl erfolgreich ausgeführt.")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen des Befehls: {e}")
    except FileNotFoundError as e:
        print(f"Fehler: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()
