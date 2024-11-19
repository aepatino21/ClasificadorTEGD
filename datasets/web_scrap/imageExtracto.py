from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By

import pandas as pd

driver = webdriver.Firefox()

driver.get('https://www.istockphoto.com/es/search/2/image?numberofpeople=one&phrase=Indigena%20amazonia')

limite = driver.find_element(By.XPATH, '/html/body/div[2]/section/div/main/div/div/div[2]/div/section/span').text

lista_fotos = []

for i in range(1, int(limite)): 
    driver.get(f'https://www.istockphoto.com/es/search/2/image?numberofpeople=one&phrase=Indigena%20amazonia&page={i}')

    WebDriverWait(driver, 5)\
        .until(expected_conditions.element_to_be_clickable((By.XPATH, "/html/body/div[2]/section/div/main/div/div/div[2]/div/div[3]")))\

    fotos = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='gallery-mosaic-asset']")

    for foto in fotos: 
        figure = foto.find_element(By.XPATH, './article/a/figure')   
        childs = figure.find_elements(By.XPATH, './child::*')
        if childs[1].tag_name == 'picture':
            img = figure.find_element(By.XPATH, './picture/img')
            alt_text = img.get_attribute('alt').split(sep='-')[0].lower()
            print(alt_text)
            keywords = ['hombre', 'mujer', 'niño', 'niña', 'anciano', 'chamán', 'jefe', 'joven']
            not_keywords = ['ilustraciones']
            if any(keyword in alt_text for keyword in keywords) and not any(not_keyword in alt_text for not_keyword in not_keywords) :
                lista_fotos.append([alt_text, img.get_attribute('src')])

driver.close()

# Se pasa el arreglo de links y descripciones de las fotos a un archivo .csv

df_fotos = pd.DataFrame(lista_fotos, columns=['Descripción', 'Link'])

df_fotos.to_csv('FotosIndigenas.csv')