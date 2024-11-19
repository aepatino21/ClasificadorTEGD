from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By

import pandas as pd

driver = webdriver.Firefox()

driver.get('https://www.shutterstock.com/es/search/indigenas-de-la-amazonia')

# limite = int(driver.find_element(By.XPATH, '/html/body/div[2]/section/div/main/div/div/div[4]/div/section/span').text)

# lista_fotos = []

# for i in range(1, limite): 
#     driver.get(f'https://www.gettyimages.es/search/2/image?phrase=indigenas%20de%20la%20amazonia&sort=mostpopular&license=rf%2Crm&page={i}')

#     container = driver.find_element(By.XPATH, '/html/body/div[2]/section/div/main/div/div/div[4]/div/div[2]')

#     fotos = container.find_elements(By.CSS_SELECTOR, "div[data-testid='galleryMosaicAsset']")

#     for foto in fotos: 
#         img = foto.find_element(By.XPATH, './article/a/figure/picture/img')
#         alt_text = img.get_attribute('alt').split(sep='-')[0].lower()
#         keywords = ['hombre', 'mujer', 'niño', 'niña', 'anciano', 'chamán', 'jefe', 'joven']
#         not_keywords = ['ilustraciones']
#         if any(keyword in alt_text for keyword in keywords) and not any(not_keyword in alt_text for not_keyword in not_keywords) :
#             lista_fotos.append([alt_text, img.get_attribute('src')])

# driver.close()

# df_fotos = pd.DataFrame(lista_fotos, columns=['Descripción', 'Link'])

# df_fotos.to_csv('FotosIndigenas2.csv')