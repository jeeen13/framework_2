import subprocess
import os


def main():
    # Pfad zur render_agent.py
    script_path = os.path.join("ns_policies", "SCoBOts_framework", "render_agent.py")

    # Befehl und Argumente
    command = ["python", script_path, "-g", "Pong", "-s", "0", "-r", "human", "-p", "default"]

    # Versuch, den Befehl auszuführen
    try:
        subprocess.run(command, check=True)
        print("Befehl erfolgreich ausgeführt.")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen des Befehls: {e}")
    except FileNotFoundError:
        print("Python-Interpreter oder render_agent.py wurde nicht gefunden.")


# Der Entry-Point der Anwendung
if __name__ == "__main__":
    main()
