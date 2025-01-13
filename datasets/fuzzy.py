import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

# Cargar tu modelo preentrenado
model = load_model('second_modelv3.keras')

# Función para obtener características de la imagen
def get_image_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    return decode_predictions(preds, top=10)[0]


# Cargar la imagen y obtener características
img_path = 'ruta_a_tu_imagen.jpg'
features = get_image_features(img_path)

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

# Crear el sistema de control difuso
ethnic_ctrl = ctrl.ControlSystem([akawayo_rule, karina_rule, arawak_rule, enepa_rule, mapoyo_rule, yabarana_rule, jivi_rule, jodi_rule, pemon_rule, puinave_rule, piaroa_rule, warao_rule, yanomami_rule, yekwana_rule])
ethnic_sim = ctrl.ControlSystemSimulation(ethnic_ctrl)

# Asignar valores a las variables difusas basadas en las características de la imagen
ethnic_sim.input['pendant'] = 1 if 'pendant' in [f[1] for f in features] else 0
ethnic_sim.input['corporal_paint'] = 1 if 'corporal_paint' in [f[1] for f in features] else 0
ethnic_sim.input['face_paint'] = 1 if 'face_paint' in [f[1] for f in features] else 0
ethnic_sim.input['modern_clothing'] = 1 if 'modern_clothing' in [f[1] for f in features] else 0
ethnic_sim.input['creole_clothing'] = 1 if 'creole_clothing' in [f[1] for f in features] else 0
ethnic_sim.input['ancestral_clothing'] = 1 if 'ancestral_clothing' in [f[1] for f in features] else 0
ethnic_sim.input['animal_fur'] = 1 if 'animal_fur' in [f[1] for f in features] else 0
ethnic_sim.input['feathers'] = 1 if 'feathers' in [f[1] for f in features] else 0
ethnic_sim.input['hat'] = 1 if 'hat' in [f[1] for f in features] else 0
ethnic_sim.input['nose_piercing'] = 1 if 'nose_piercing' in [f[1] for f in features] else 0
ethnic_sim.input['bowl_cut'] = 1 if 'bowl_cut' in [f[1] for f in features] else 0
ethnic_sim.input['tendrils'] = 1 if 'tendrils' in [f[1] for f in features] else 0
ethnic_sim.input['arm_accesory'] = 1 if 'arm_accesory' in [f[1] for f in features] else 0
ethnic_sim.input['bracelets'] = 1 if 'bracelets' in [f[1] for f in features] else 0

# Ejecutar la simulación
ethnic_sim.compute()

# Mostrar los resultados de pertenencia
print('Akawayo:', ethnic_sim.output['akawayo'])
print('Karina:', ethnic_sim.output['karina'])
print('Arawak:', ethnic_sim.output['arawak'])
print('Enepa:', ethnic_sim.output['enepa'])
print('Mapoyo:', ethnic_sim.output['mapoyo'])
print('Yabarana:', ethnic_sim.output['yabarana'])
print('Jivi:', ethnic_sim.output['jivi'])
print('Jodi:', ethnic_sim.output['jodi'])
print('Pemon:', ethnic_sim.output['pemon'])
print('Puinave:', ethnic_sim.output['puinave'])
print('Piaroa:', ethnic_sim.output['piaroa'])
print('Warao:', ethnic_sim.output['warao'])
print('Yanomami:', ethnic_sim.output['yanomami'])
print('Yekwana:', ethnic_sim.output['yekwana'])
