import undetected_chromedriver as uc
import pandas as pd
import time
import sys

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python run_script.py <numero de página>")
        sys.exit(1)

    argument = sys.argv[1]

    driver = uc.Chrome()
    # Hay que acceder 2 veces a la pagina o si no muestra las imágenes de la primera página sin importar el número
    driver.get(f'https://www.shutterstock.com/es/search/indigenas-amazonia?image_type=photo&page={argument}')
    driver.get(f'https://www.shutterstock.com/es/search/indigenas-amazonia?image_type=photo&page={argument}')


    limit = int(driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/div/div/div[1]/div/div[7]/div[2]/div[2]/div[2]/div/span').text.split()[-1])
    
    WebDriverWait(driver, 5)\
        .until(expected_conditions.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[3]/div/div/div[1]/div/div[7]/div[1]/div[2]')))

        
    # Esta parte sirve para ir bajando la página, lo que hace que carguen las imágenes
    inner_height = driver.execute_script('return window.innerHeight')

    while True:
        driver.execute_script('window.scrollTo(0, window.scrollY + 200)')
        time.sleep(0.20)
        current_height = driver.execute_script('return window.scrollY')
        total_height = driver.execute_script('return document.body.scrollHeight')
        if current_height >= total_height - inner_height - 200:
            break
        

    fotos = driver.find_elements(By.CSS_SELECTOR, "div[data-automation='AssetGrids_GridItemContainer_div']")

    photo_list = []

    for foto in fotos:
        childs = foto.find_elements(By.XPATH, './*')
        if len(childs) == 3:
            img = foto.find_element(By.XPATH, './div[1]/div/picture/img')
        elif len(childs) == 4: 
            img = foto.find_element(By.XPATH, './div[2]/div/picture/img')
        alt_text = img.get_attribute('title').lower()
        src_link = img.get_attribute('src')
        keywords = ['hombre', 'mujer', 'niño', 'niña', 'anciano', 'chamán', 'jefe', 'joven']
        if any(keyword in alt_text for keyword in keywords):
            photo_list.append([alt_text, src_link])

    driver.quit()

    print(photo_list)

    # Se pasa el arreglo de links y descripciones de las fotos a un archivo .csv

    df_fotos = pd.DataFrame(photo_list, columns=['Descripción', 'Link'])

    df_fotos.to_csv(f'FotosIndigenas{argument}.csv')