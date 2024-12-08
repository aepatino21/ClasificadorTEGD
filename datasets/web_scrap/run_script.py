import subprocess
import sys

if len(sys.argv) < 2:
    print("Uso: python run_script.py <numero de página>")
    sys.exit(1)

argument = sys.argv[1]

# Instalar las dependencias
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Ejecutar el script principal
subprocess.check_call([sys.executable, "extractShutter.py", argument])