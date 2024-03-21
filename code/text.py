from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
driver = webdriver.Edge()

driver.get('https://www.tripadvisor.com/Attractions-g191-Activities-oa0-United_States.html')

#element = driver.find_element(By.ID, 'sb_form_q')
#element.send_keys('WebDriver')
#clearelement.submit()

soup = BeautifulSoup(driver.page_source, 'html.parser')
time.sleep(5)


imgs_items = soup.find_all('ul', attrs={"class":"zHxHb"})

img_item_list = [a.find_all('img') for a in imgs_items]
image_links = [img['src'] for img in img_item_list[0]]



print(image_links)

 



list_items = soup.find_all('div', attrs={"class":"hZuqH y"})
titre_item = soup.find_all('div', attrs={"class":"XfVdV o AIbhI"})
lieu_item = soup.find_all('div', {'class':'bRMrl _Y K'})

span_item = soup.find_all('span', class_='jXmYW')

#next_div_biGQs = span_item[0].find_next('div', class_='biGQs')


lieu_item_list = [a.getText().strip() for a in lieu_item]




print(lieu_item_list[0])
print("------")

#print(next_div_biGQs.getText().strip())



driver.quit()