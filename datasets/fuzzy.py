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

# Cargar la tabla atributo-valor en un DataFrame de pandas
df = pd.DataFrame(data)

# Verificar valores únicos para asegurar que están dentro del rango esperado
print("Valores únicos en corporal_paint:", df['corporal_paint'].unique())

# Rellenar valores nulos con un valor predeterminado (0 en este caso)
df.fillna(0, inplace=True)

# Asegurarse de que todos los valores estén en el rango 0-1
df = df[(df['corporal_paint'].isin([0, 1])) &
        (df['pendant'].isin([0, 1])) &
        (df['face_paint'].isin([0, 1])) &
        (df['modern_clothing'].isin([0, 1])) &
        (df['creole_clothing'].isin([0, 1])) &
        (df['ancestral_clothing'].isin([0, 1])) &
        (df['animal_fur'].isin([0, 1])) &
        (df['feathers'].isin([0, 1])) &
        (df['hat'].isin([0, 1])) &
        (df['nose_piercing'].isin([0, 1])) &
        (df['bowl_cut'].isin([0, 1])) &
        (df['tendrils'].isin([0, 1])) &
        (df['arm_accesory'].isin([0, 1])) &
        (df['bracelets'].isin([0, 1]))]

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

# Definir funciones de pertenencia
pendant.automf(3)
corporal_paint.automf(3)
face_paint.automf(3)
modern_clothing.automf(3)
creole_clothing.automf(3)
ancestral_clothing.automf(3)
animal_fur.automf(3)
feathers.automf(3)
hat.automf(3)
nose_piercing.automf(3)
bowl_cut.automf(3)
tendrils.automf(3)
arm_accesory.automf(3)
bracelets.automf(3)

akawayo.automf(3)
karina.automf(3)
arawak.automf(3)
enepa.automf(3)
mapoyo.automf(3)
yabarana.automf(3)
jivi.automf(3)
jodi.automf(3)
pemon.automf(3)
puinave.automf(3)
piaroa.automf(3)
warao.automf(3)
yanomami.automf(3)
yekwana.automf(3)

# Crear reglas difusas (a modo de ejemplo)
rule1 = ctrl.Rule(pendant['good'] & ancestral_clothing['good'], akawayo['good'])
rule2 = ctrl.Rule(face_paint['poor'] & modern_clothing['good'], karina['good'])
# Añade más reglas según la lógica de tu modelo

# Crear el sistema de control difuso
ethnic_ctrl = ctrl.ControlSystem([rule1, rule2])
ethnic_sim = ctrl.ControlSystemSimulation(ethnic_ctrl)

for index, row in df.iterrows():
    try:
        ethnic_sim.input['pendant'] = row['pendant']
        ethnic_sim.input['corporal_paint'] = row['corporal_paint']
        ethnic_sim.input['face_paint'] = row['face_paint']
        ethnic_sim.input['modern_clothing'] = row['modern_clothing']
        ethnic_sim.input['creole_clothing'] = row['creole_clothing']
        ethnic_sim.input['ancestral_clothing'] = row['ancestral_clothing']
        ethnic_sim.input['animal_fur'] = row['animal_fur']
        ethnic_sim.input['feathers'] = row['feathers']
        ethnic_sim.input['hat'] = row['hat']
        ethnic_sim.input['nose_piercing'] = row['nose_piercing']
        ethnic_sim.input['bowl_cut'] = row['bowl_cut']
        ethnic_sim.input['tendrils'] = row['tendrils']
        ethnic_sim.input['arm_accesory'] = row['arm_accesory']
        ethnic_sim.input['bracelets'] = row['bracelets']

        ethnic_sim.compute()

        print(f"Etnia predicha para la fila {index}:")
        print(f"Akawayo: {ethnic_sim.output['akawayo']}")
        print(f"Karina: {ethnic_sim.output['karina']}")
        print(f"Arawak: {ethnic_sim.output['arawak']}")
        print(f"E'ñepa: {ethnic_sim.output['enepa']}")
        print(f"Mapoyo: {ethnic_sim.output['mapoyo']}")
        print(f"Yabarana: {ethnic_sim.output['yabarana']}")
        print(f"Jivi: {ethnic_sim.output['jivi']}")
        print(f"Jodi: {ethnic_sim.output['jodi']}")
        print(f"Pemón: {ethnic_sim.output['pemon']}")
        print(f"Puinave: {ethnic_sim.output['puinave']}")
        print(f"Piaroa: {ethnic_sim.output['piaroa']}")
        print(f"Warao: {ethnic_sim.output['warao']}")
        print(f"Yanomami: {ethnic_sim.output['yanomami']}")
        print(f"Ye'kwana: {ethnic_sim.output['yekwana']}")
        print("")
    except ValueError as e:
        print(f"Error en la fila {index}: {e}")

# Para obtener una visualización más clara de los resultados, podrías considerar guardar los resultados en un DataFrame
predicted_ethnicities = {
    'akawayo': [],
    'karina': [],
    'arawak': [],
    'enepa': [],
    'mapoyo': [],
    'yabarana': [],
    'jivi': [],
    'jodi': [],
    'pemon': [],
    'puinave': [],
    'piaroa': [],
    'warao': [],
    'yanomami': [],
    'yekwana': []
}

for index, row in df.iterrows():
    try:
        ethnic_sim.input['pendant'] = row['pendant']
        ethnic_sim.input['corporal_paint'] = row['corporal_paint']
        ethnic_sim.input['face_paint'] = row['face_paint']
        ethnic_sim.input['modern_clothing'] = row['modern_clothing']
        ethnic_sim.input['creole_clothing'] = row['creole_clothing']
        ethnic_sim.input['ancestral_clothing'] = row['ancestral_clothing']
        ethnic_sim.input['animal_fur'] = row['animal_fur']
        ethnic_sim.input['feathers'] = row['feathers']
        ethnic_sim.input['hat'] = row['hat']
        ethnic_sim.input['nose_piercing'] = row['nose_piercing']
        ethnic_sim.input['bowl_cut'] = row['bowl_cut']
        ethnic_sim.input['tendrils'] = row['tendrils']
        ethnic_sim.input['arm_accesory'] = row['arm_accesory']
        ethnic_sim.input['bracelets'] = row['bracelets']

        ethnic_sim.compute()

        predicted_ethnicities['akawayo'].append(ethnic_sim.output['akawayo'])
        predicted_ethnicities['karina'].append(ethnic_sim.output['karina'])
        predicted_ethnicities['arawak'].append(ethnic_sim.output['arawak'])
        predicted_ethnicities['enepa'].append(ethnic_sim.output['enepa'])
        predicted_ethnicities['mapoyo'].append(ethnic_sim.output['mapoyo'])
        predicted_ethnicities['yabarana'].append(ethnic_sim.output['yabarana'])
        predicted_ethnicities['jivi'].append(ethnic_sim.output['jivi'])
        predicted_ethnicities['jodi'].append(ethnic_sim.output['jodi'])
        predicted_ethnicities['pemon'].append(ethnic_sim.output['pemon'])
        predicted_ethnicities['puinave'].append(ethnic_sim.output['puinave'])
        predicted_ethnicities['piaroa'].append(ethnic_sim.output['piaroa'])
        predicted_ethnicities['warao'].append(ethnic_sim.output['warao'])
        predicted_ethnicities['yanomami'].append(ethnic_sim.output['yanomami'])
        predicted_ethnicities['yekwana'].append(ethnic_sim.output['yekwana'])
    except ValueError as e:
        print(f"Error en la fila {index}: {e}")

# Convertir los resultados predichos en un DataFrame
df_predicciones = pd.DataFrame(predicted_ethnicities)

# Guardar las predicciones en un archivo CSV
df_predicciones.to_csv('predicciones_etnias.csv', index=False)
print("Las predicciones se han guardado en 'predicciones_etnias.csv'")

