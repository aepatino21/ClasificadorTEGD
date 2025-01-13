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
def get_image_features(img_path):
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
    features_dict = {name: pred for name, pred in zip(feature_names, preds[0])}
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

# Importar las reglas desde tu archivo
# Regla difusa: Akawayo
akawayo_rule = ctrl.Rule(pendant['good'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['poor'] & creole_clothing['good'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['good'] & hat['good'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], akawayo['good'])

# Regla difusa: Karina
karina_rule = ctrl.Rule(pendant['poor'] & corporal_paint['poor'] & face_paint['good'] & modern_clothing['poor'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['good'], karina['good'])

# Regla difusa: Arawak
arawak_rule = ctrl.Rule(pendant['good'] & corporal_paint['poor'] & face_paint['good'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['good'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['good'], arawak['good'])

# Regla difusa: E'nepa
enepa_rule = ctrl.Rule(pendant['good'] & corporal_paint['poor'] & face_paint['good'] & modern_clothing['poor'] & creole_clothing['good'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['good'], enepa['good'])

# Regla difusa: Mapoyo
mapoyo_rule = ctrl.Rule(pendant['good'] & corporal_paint['good'] & face_paint['poor'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['good'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['good'] & bracelets['good'], mapoyo['good'])

# Regla difusa: Yabarana
yabarana_rule = ctrl.Rule(pendant['poor'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['good'] & creole_clothing['good'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], yabarana['good'])

# Regla difusa: Jivi
jivi_rule = ctrl.Rule(pendant['poor'] & corporal_paint['poor'] & face_paint['good'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['poor'] & animal_fur['poor'] & feathers['poor'] & hat['good'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], jivi['good'])

# Regla difusa: Jodi
jodi_rule = ctrl.Rule(pendant['good'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['poor'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], jodi['good'])

# Regla difusa: Pemon
pemon_rule = ctrl.Rule(pendant['good'] & corporal_paint['good'] & face_paint['good'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], pemon['good'])

# Regla difusa: Puinave
puinave_rule = ctrl.Rule(pendant['poor'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['poor'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], puinave['good'])

# Regla difusa: Piaroa
piaroa_rule = ctrl.Rule(pendant['good'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], piaroa['good'])

# Regla difusa: Warao
warao_rule = ctrl.Rule(pendant['good'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], warao['good'])

# Regla difusa: Yanomami
yanomami_rule = ctrl.Rule(pendant['good'] & corporal_paint['good'] & face_paint['poor'] & modern_clothing['poor'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['good'] & bowl_cut['good'] & tendrils['good'] & arm_accesory['good'] & bracelets['poor'], yanomami['good'])

# Regla difusa: Ye'kwana-Sanoma
yekwana_rule = ctrl.Rule(pendant['poor'] & corporal_paint['poor'] & face_paint['poor'] & modern_clothing['good'] & creole_clothing['poor'] & ancestral_clothing['good'] & animal_fur['poor'] & feathers['poor'] & hat['poor'] & nose_piercing['poor'] & bowl_cut['poor'] & tendrils['poor'] & arm_accesory['poor'] & bracelets['poor'], yekwana['good'])

# Crear el sistema de control difuso con las reglas importadas
ethnic_ctrl = ctrl.ControlSystem([akawayo_rule, karina_rule, arawak_rule, enepa_rule, mapoyo_rule, yabarana_rule, jivi_rule, jodi_rule, pemon_rule, puinave_rule, piaroa_rule, warao_rule, yanomami_rule, yekwana_rule])
ethnic_sim = ctrl.ControlSystemSimulation(ethnic_ctrl)

# Usar las características obtenidas de la imagen para las variables difusas desde el diccionario
ethnic_sim.input['pendant'] = features_dict.get('pendant', 0)
ethnic_sim.input['corporal_paint'] = features_dict.get('corporal_paint', 0)
ethnic_sim.input['face_paint'] = features_dict.get('face_paint', 0)
ethnic_sim.input['modern_clothing'] = features_dict.get('modern_clothing', 0)
ethnic_sim.input['creole_clothing'] = features_dict.get('creole_clothing', 0)
ethnic_sim.input['ancestral_clothing'] = features_dict.get('ancestral_clothing', 0)
ethnic_sim.input['animal_fur'] = features_dict.get('animal_fur', 0)
ethnic_sim.input['feathers'] = features_dict.get('feathers', 0)
ethnic_sim.input['hat'] = features_dict.get('hat', 0)
ethnic_sim.input['nose_piercing'] = features_dict.get('nose_piercing', 0)
ethnic_sim.input['bowl_cut'] = features_dict.get('bowl_cut', 0)
ethnic_sim.input['tendrils'] = features_dict.get('tendrils', 0)
ethnic_sim.input['arm_accesory'] = features_dict.get('arm_accesory', 0)
ethnic_sim.input['bracelets'] = features_dict.get('bracelets', 0)

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
