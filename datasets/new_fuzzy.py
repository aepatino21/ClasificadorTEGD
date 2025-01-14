import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

# Cargar el modelo preentrenado
model = load_model('second_modelv69_definitive.keras')

# Función para obtener características de la imagen
def get_image_features(img_path, threshold=0.50):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)

    # Suponemos que el modelo devuelve un diccionario con los nombres de las características
    feature_names = ['pendant', 'corporal_paint', 'face_paint', 'modern_clothing', 'creole_clothing',
                     'ancestral_clothing', 'animal_fur', 'feathers', 'hat', 'nose_piercing',
                     'bowl_cut', 'tendrils', 'arm_accesory', 'bracelets']

    # Crear un diccionario con los nombres de las características
    features_dict = {name: (1 if pred >= threshold else 0) for name, pred in zip(feature_names, preds[0])}
    return features_dict

# Cargar la imagen y obtener características
img_path = '../images/1.jpeg'
features_dict = get_image_features(img_path)

# Definir las variables difusas y las reglas
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

# Definir funciones de pertenencia binarias
for antecedent in [pendant, corporal_paint, face_paint, modern_clothing, creole_clothing, ancestral_clothing, animal_fur, feathers, hat, nose_piercing, bowl_cut, tendrils, arm_accesory, bracelets]:
    antecedent['absent'] = fuzz.trimf(antecedent.universe, [0, 0, 0.5])
    antecedent['present'] = fuzz.trimf(antecedent.universe, [0.5, 1, 1])

# Definir funciones de pertenencia de salida
for consequent in [akawayo, karina, arawak, enepa, mapoyo, yabarana, jivi, jodi, pemon, puinave, piaroa, warao, yanomami, yekwana]:
    consequent['low'] = fuzz.trimf(consequent.universe, [0, 0, 50])
    consequent['medium'] = fuzz.trimf(consequent.universe, [0, 50, 100])
    consequent['high'] = fuzz.trimf(consequent.universe, [50, 100, 100])

# Importar las reglas desde tu archivo
# Regla difusa: Akawayo

akawayo_rule = ctrl.Rule(pendant['present'] | face_paint['present'] | creole_clothing['present'] | hat['present'], akawayo['high'])

# Regla difusa: Karina
karina_rule = ctrl.Rule(pendant['absent'] | corporal_paint['absent'] | face_paint['present'] | modern_clothing['absent'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['present'], karina['high'])

# Regla difusa: Arawak
arawak_rule = ctrl.Rule(pendant['present'] | corporal_paint['absent'] | face_paint['present'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['present'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['present'], arawak['high'])

# Regla difusa: E'nepa
enepa_rule = ctrl.Rule(pendant['present'] | corporal_paint['absent'] | face_paint['present'] | modern_clothing['absent'] | creole_clothing['present'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['present'], enepa['high'])

# Regla difusa: Mapoyo
mapoyo_rule = ctrl.Rule(pendant['present'] | corporal_paint['present'] | face_paint['absent'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['present'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['present'] | bracelets['present'], mapoyo['high'])

# Regla difusa: Yabarana
yabarana_rule = ctrl.Rule(pendant['absent'] | corporal_paint['absent'] | face_paint['absent'] | modern_clothing['present'] | creole_clothing['present'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], yabarana['high'])

# Regla difusa: Jivi
jivi_rule = ctrl.Rule(pendant['absent'] | corporal_paint['absent'] | face_paint['present'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['absent'] | animal_fur['absent'] | feathers['absent'] | hat['present'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], jivi['high'])

# Regla difusa: Jodi
jodi_rule = ctrl.Rule(pendant['present'] | corporal_paint['absent'] | face_paint['absent'] | modern_clothing['absent'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], jodi['high'])

# Regla difusa: Pemon
pemon_rule = ctrl.Rule(pendant['present'] | corporal_paint['present'] | face_paint['present'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], pemon['high'])

# Regla difusa: Puinave
puinave_rule = ctrl.Rule(pendant['absent'] | corporal_paint['absent'] | face_paint['absent'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['absent'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], puinave['high'])

# Regla difusa: Piaroa
piaroa_rule = ctrl.Rule(pendant['present'] | corporal_paint['absent'] | face_paint['absent'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], piaroa['high'])

# Regla difusa: Warao
warao_rule = ctrl.Rule(pendant['present'] | corporal_paint['absent'] | face_paint['absent'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], warao['high'])

# Regla difusa: Yanomami
yanomami_rule = ctrl.Rule(pendant['present'] | corporal_paint['present'] | face_paint['absent'] | modern_clothing['absent'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['present'] | bowl_cut['present'] | tendrils['present'] | arm_accesory['present'] | bracelets['absent'], yanomami['high'])

# Regla difusa: Ye'kwana-Sanoma
yekwana_rule = ctrl.Rule(pendant['absent'] | corporal_paint['absent'] | face_paint['absent'] | modern_clothing['present'] | creole_clothing['absent'] | ancestral_clothing['present'] | animal_fur['absent'] | feathers['absent'] | hat['absent'] | nose_piercing['absent'] | bowl_cut['absent'] | tendrils['absent'] | arm_accesory['absent'] | bracelets['absent'], yekwana['high'])

# Crear el sistema de control difuso con las reglas importadas
ethnic_ctrl = ctrl.ControlSystem([akawayo_rule, karina_rule, arawak_rule, enepa_rule, mapoyo_rule, yabarana_rule, jivi_rule, jodi_rule, pemon_rule, puinave_rule, piaroa_rule, warao_rule, yanomami_rule, yekwana_rule])
ethnic_sim = ctrl.ControlSystemSimulation(ethnic_ctrl)

# Usar las características obtenidas de la imagen para las variables difusas desde el diccionario
default_value = 0
ethnic_sim.input['pendant'] = features_dict.get('pendant', default_value)
ethnic_sim.input['corporal_paint'] = features_dict.get('corporal_paint', default_value)
ethnic_sim.input['face_paint'] = features_dict.get('face_paint', default_value)
ethnic_sim.input['modern_clothing'] = features_dict.get('modern_clothing', default_value)
ethnic_sim.input['creole_clothing'] = features_dict.get('creole_clothing', default_value)
ethnic_sim.input['ancestral_clothing'] = features_dict.get('ancestral_clothing', default_value)
ethnic_sim.input['animal_fur'] = features_dict.get('animal_fur', default_value)
ethnic_sim.input['feathers'] = features_dict.get('feathers', default_value)
ethnic_sim.input['hat'] = features_dict.get('hat', default_value)
ethnic_sim.input['nose_piercing'] = features_dict.get('nose_piercing', default_value)
ethnic_sim.input['bowl_cut'] = features_dict.get('bowl_cut', default_value)
ethnic_sim.input['tendrils'] = features_dict.get('tendrils', default_value)
ethnic_sim.input['arm_accesory'] = features_dict.get('arm_accesory', default_value)
ethnic_sim.input['bracelets'] = features_dict.get('bracelets', default_value)

# Computar las predicciones
ethnic_sim.compute()

# Mostrar los resultados
print("Predicciones basadas en la imagen:")
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
