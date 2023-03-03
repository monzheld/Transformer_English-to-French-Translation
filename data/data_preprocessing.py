import os
import urllib3
import zipfile
import shutil
import pandas as pd

# 데이터셋 로드 
http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
user_agent = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
with http.request('GET', url, preload_content=False, headers=user_agent) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall()

os.remove(zipfilename)
about_txt = os.path.join(path, '_about.txt')
os.remove(about_txt)

fra_eng_csv = pd.read_csv('fra.txt', names=['ENG', 'FRA', 'lic'], sep='\t')
del fra_eng_csv['lic']
print('데이터셋 크기:', fra_eng_csv.shape)
print('데이터셋 예시:')
print(fra_eng_csv.head())


