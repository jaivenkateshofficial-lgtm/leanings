import threading
import requests
from bs4 import BeautifulSoup

urls=['https://www.infineon.com/product-information/automotive-cybersecurity?uid=ci33590001&aid=ai3359&gclsrc=aw.ds&gad_source=1&gad_campaignid=12750974848&gbraid=0AAAAADpmf9fOqHnyOiQdCnyx0zoDtqOgr&gclid=EAIaIQobChMI0vPl_Y3RkAMV8jyDAx0jwxVNEAAYASAAEgJCdfD_BwE',
'https://kpitindia.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/learn/lecture/44469918#learning-tools']
def fetch_content(url):
    response= requests.get(url)
    soup=BeautifulSoup(response.content,'html.parser')
    print(f'length:{len(soup.text)} from {url}')
    print(soup.text.encode('utf-8'))
threads=[]
for url in urls:
    thread=threading.Thread(target=fetch_content,args=(url,))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()

print('all web pages fetched')