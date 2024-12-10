import subprocess
import sys

if len(sys.argv) < 2:
    print("Uso: python run_script.py <numero de página>")
    sys.exit(1)

start_page = int(sys.argv[1])

# Instalar las dependencias
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Ejecutar el script principal
for i in range (start_page, 182):
    subprocess.check_call([sys.executable, "extractShutter.py", str(i)])    