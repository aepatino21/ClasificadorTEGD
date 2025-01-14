import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import json

# Cargar el modelo preentrenado
model = load_model('second_modelv69_definitive.keras')

# Función para obtener características de la imagen
def get_image_features(img_path, threshold=0.50):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)

    feature_names = ['pendant', 'corporal_paint', 'face_paint', 'modern_clothing', 'creole_clothing',
                     'ancestral_clothing', 'animal_fur', 'feathers', 'hat', 'nose_piercing',
                     'bowl_cut', 'tendrils', 'arm_accesory', 'bracelets']

    # Crear un diccionario con los nombres de las características
    features_dict = {name: (1 if pred >= threshold else 0) for name, pred in zip(feature_names, preds[0])}
    return features_dict

# Cargar la imagen y obtener características
img_path = '../images/2.jpg'
features_dict = get_image_features(img_path)
print(features_dict)

# Contar las características importantes presentes para Akawayo
important_akawayo_features = ['pendant', 'face_paint', 'creole_clothing', 'hat']
akawayo_feature_count = sum(features_dict.get(feature, 0) for feature in important_akawayo_features)

# Contar las características importantes presentes para Karina
important_karina_features = ['ancestral_clothing', 'face_paint', 'bracelets']
karina_feature_count = sum(features_dict.get(feature, 0) for feature in important_karina_features)

# Contar las caracteristicas importantes presentes para Arawak
important_arawak_features = ['pendant', 'face_paint', 'modern_clothing', 'ancestral_clothing', 'hat', 'bracelets']
arawak_feature_count = sum(features_dict.get(feature, 0) for feature in important_arawak_features)

# Contar las caracteristicas importantes presentes para E'ñepa
important_enepa_features = ['pendant', 'face_paint', 'creole_clothing', 'ancestral_clothing', 'bracelets']
enepa_feature_count = sum(features_dict.get(feature, 0) for feature in important_enepa_features)

# Contar las caracteristicas importantes presentes para Mapoyo
important_mapoyo_features = ['pendant', 'corporal_paint', 'modern_clothing', 'ancestral_clothing', 'animal_fur', 'arm_accesory', 'bracelets']
mapoyo_feature_count = sum(features_dict.get(feature, 0) for feature in important_mapoyo_features)

# Contar las caracteristicas importantes presentes para Yabarana
important_yabarana_features = ['modern_clothing', 'creole_clothing', 'ancestral_clothing']
yabarana_feature_count = sum(features_dict.get(feature, 0) for feature in important_yabarana_features)

# Contar las caracteristicas importantes presentes para Jivi
important_jivi_features = ['face_paint', 'modern_clothing', 'hat']
jivi_feature_count = sum(features_dict.get(feature, 0) for feature in important_jivi_features)

# Contar las caracteristicas importantes presentes para Jodi
important_jodi_features = ['pendant', 'ancestral_clothing']
jodi_feature_count = sum(features_dict.get(feature, 0) for feature in important_jodi_features)

# Contar las caracteristicas importantes presentes para Pemón
important_pemon_features = ['pendant', 'corporal_paint', 'face_paint', 'modern_clothing', 'ancestral_clothing']
pemon_feature_count = sum(features_dict.get(feature, 0) for feature in important_pemon_features)

# Contar las caracteristicas importantes presentes para Puinave
important_puinave_features = ['modern_clothing']
puinave_feature_count = sum(features_dict.get(feature, 0) for feature in important_puinave_features)

# Contar las caracteristicas importantes presentes para Piaroa
important_piaroa_features = ['pendant', 'modern_clothing', 'ancestral_clothing']
piaroa_feature_count = sum(features_dict.get(feature, 0) for feature in important_piaroa_features)

# Contar las caracteristicas importantes presentes para Warao
important_warao_features = ['pendant', 'modern_clothing', 'ancestral_clothing']
warao_feature_count = sum(features_dict.get(feature, 0) for feature in important_warao_features)

# Contar las caracteristicas importantes presentes para Yanomami
important_yanomami_features = ['pendant', 'corporal_paint', 'ancestral_clothing', 'nose_piercing', 'bowl_cut', 'tendrils', 'arm_accesory']
yanomami_feature_count = sum(features_dict.get(feature, 0) for feature in important_yanomami_features)

# Contar las caracteristicas importantes presentes para Ye'kwana
important_yekwana_features = ['modern_clothing', 'ancestral_clothing']
yekwana_feature_count = sum(features_dict.get(feature, 0) for feature in important_yekwana_features)

# Contar las caracteristicas importantes presentes para No Etnia
important_noetnia_features = ['modern_clothing']
noetnia_feature_count = sum(features_dict.get(feature, 0) for feature in important_noetnia_features)

 # Definir funciones para calcular el valor de pertenencia
def calculate_membership_akawayo(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 24.99
   elif count >= 2 and count <= 3:
       return 64.50
   elif count == 4:
       return 84.99

def calculate_membership_karina(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 24.99
   elif count == 2:
       return 64.50
   elif count == 3:
       return 84.99

def calculate_membership_arawak(count):
   if count == 0:
       return 1.99
   elif count >= 1 and count <= 2:
       return 30.99
   elif count >= 3 and count <= 5:
       return 68.50
   elif count == 6:
       return 83.99

def calculate_membership_enepa(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 24.99
   elif count >= 2 and count <= 4:
       return 64.50
   elif count == 5:
       return 84.99

def calculate_membership_mapoyo(count):
   if count == 0:
       return 1.99
   elif count >= 1 and count <= 2:
       return 30.99
   elif count >= 3 and count <= 6:
       return 70.50
   elif count == 7:
       return 86.99

def calculate_membership_yabarana(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 20.99
   elif count == 2:
       return 54.99
   elif count == 3:
       return 80.99

def calculate_membership_jivi(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 20.99
   elif count == 2:
       return 54.99
   elif count == 3:
       return 80.99

def calculate_membership_jodi(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 20.99
   elif count == 2:
       return 54.99
   elif count == 3:
       return 80.99

def calculate_membership_pemon(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 24.99
   elif count >= 2 and count <= 4:
       return 64.50
   elif count == 5:
       return 84.99

def calculate_membership_puinave(count):
    if count == 0:
        return 1.99
    elif count == 1:
        return 24.50

def calculate_membership_piaroa(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 20.99
   elif count == 2:
       return 54.99
   elif count == 3:
       return 80.99

def calculate_membership_warao(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 20.99
   elif count == 2:
       return 54.99
   elif count == 3:
       return 80.99

def calculate_membership_yanomami(count):
   if count == 0:
       return 1.99
   elif count >= 1 and count <= 2:
       return 30.99
   elif count >= 3 and count <= 6:
       return 70.50
   elif count == 7:
       return 86.99

def calculate_membership_yekwana(count):
   if count == 0:
       return 1.99
   elif count == 1:
       return 34.99
   elif count == 2:
       return 75.99

def calculate_membership_noetnia(count, features):
    if count == 1 and features.get('modern_clothing') == 1: # Asegurarse de que todos los demás elementos son 0
        for key, value in features.items():
            if key != 'modern_clothing' and value != 0:
                return 9.99 # Regresar 0 si cualquier otro elemento tiene un valor diferente de 0
        return 77.99 # Si todos los demás elementos son 0, regresar 77.99
    return 1.99 # Regresar 0 si la condición no se cumple


# Crear el diccionario de pertenencia
membership_dict = {
   'Akawayo': calculate_membership_akawayo(akawayo_feature_count),
   'Karina': calculate_membership_karina(karina_feature_count),
   'Arawak': calculate_membership_arawak(arawak_feature_count),
   'Enepa': calculate_membership_enepa(enepa_feature_count),
   'Mapoyo': calculate_membership_mapoyo(mapoyo_feature_count),
   'Yabarana': calculate_membership_yabarana(yabarana_feature_count),
   'Jivi': calculate_membership_jivi(jivi_feature_count),
   'Jodi': calculate_membership_jodi(jodi_feature_count),
   'Pemon': calculate_membership_pemon(pemon_feature_count),
   'Puinave': calculate_membership_puinave(puinave_feature_count),
   'Piaroa': calculate_membership_piaroa(piaroa_feature_count),
   'Warao': calculate_membership_warao(warao_feature_count),
   'Yanomami': calculate_membership_yanomami(yanomami_feature_count),
   'Yekwana': calculate_membership_yekwana(yekwana_feature_count),
   'NoEtnia': calculate_membership_noetnia(noetnia_feature_count, features_dict)
}

# Convertir el diccionario a formato JSON
membership_json = json.dumps(membership_dict, indent=4)

# Mostrar el resultado
print("Valores de pertenencia por etnia:")
print(membership_json)
