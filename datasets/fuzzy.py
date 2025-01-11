import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Crear un DataFrame con la tabla atributo-valor
data = {
    'pendant': [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    'corporal_paint': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'face_paint': [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'modern_clothing': [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    'creole_clothing': [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'ancestral_clothing': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    'animal_fur': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'feathers': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'hat': [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'nose_piercing': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'bowl_cut': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'tendrils': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'arm_accesory': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'bracelets': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Etnias': ["Akawayo", "Kari'ña", "Arawak", "E'ñepa", "Mapoyo", "Yabarana", "Jivi", "Jodi", "Pemón", "Puinave", "Piaroa", "Warao", "Yanomami", "Ye'kwana-Sanoma"]
}

# Este código carga tu tabla atributo-valor en un DataFrame de pandas para facilitar su manejo y procesamiento.
df = pd.DataFrame(data)

# LOGICA DIFUSA
# Crear las variables difusas
pendant = ctrl.Antecedent(np.arange(0, 2, 1), 'pendant')
corporal_paint = ctrl.Antecedent(np.arange(0, 2, 1), 'corporal_paint')
face_paint = ctrl.Antecedent(np.arange(0, 2, 1), 'face_paint')
modern_clothing = ctrl.Antecedent(np.arange(0, 2, 1), 'modern_clothing')
creole_clothing = ctrl.Antecedent(np.arange(0, 2, 1), 'creole_clothing')
ancestral_clothing = ctrl.Antecedent(np.arange(0, 2, 1), 'ancestral_clothing')
animal_fur = ctrl.Antecedent(np.arange(0, 2, 1), 'animal_fur')
feathers = ctrl.Antecedent(np.arange(0, 2, 1), 'feathers')
hat = ctrl.Antecedent(np.arange(0, 2, 1), 'hat')
nose_piercing = ctrl.Antecedent(np.arange(0, 2, 1), 'nose_piercing')
bowl_cut = ctrl.Antecedent(np.arange(0, 2, 1), 'bowl_cut')
tendrils = ctrl.Antecedent(np.arange(0, 2, 1), 'tendrils')
arm_accesory = ctrl.Antecedent(np.arange(0, 2, 1), 'arm_accesory')
bracelets = ctrl.Antecedent(np.arange(0, 2, 1), 'bracelets')

# Crear las salidas difusas para cada etnia
akawayo = ctrl.Consequent(np.arange(0, 101, 1), 'akawayo')
karina = ctrl.Consequent(np.arange(0, 101, 1), 'karina')
arawak = ctrl.Consequent(np.arange(0, 101, 1), 'arawak')
enepa = ctrl.Consequent(np.arange(0, 101, 1), 'enepa')
mapoyo = ctrl.Consequent(np.arange(0, 101, 1), 'mapoyo')
yabarana = ctrl.Consequent(np.arange(0, 101, 1), 'yabarana')
jivi = ctrl.Consequent(np.arange(0, 101, 1), 'jivi')
jodi = ctrl.Consequent(np.arange(0, 101, 1), 'jodi')
pemon = ctrl.Consequent(np.arange(0, 101, 1), 'pemon')
puinave = ctrl.Consequent(np.arange(0, 101, 1), 'puinave')
piaroa = ctrl.Consequent(np.arange(0, 101, 1), 'piaroa')
warao = ctrl.Consequent(np.arange(0, 101, 1), 'warao')
yanomami = ctrl.Consequent(np.arange(0, 101, 1), 'yanomami')
yekwana = ctrl.Consequent(np.arange(0, 101, 1), 'yekwana')

# Definir las funciones de membresía para las variables difusas
for etnia in [akawayo, karina, arawak, enepa, mapoyo, yabarana, jivi, jodi, pemon, puinave, piaroa, warao, yanomami, yekwana]:
    etnia['low'] = fuzz.trimf(etnia.universe, [0, 0, 50])
    etnia['medium'] = fuzz.trimf(etnia.universe, [0, 50, 100])
    etnia['high'] = fuzz.trimf(etnia.universe, [50, 100, 100])
