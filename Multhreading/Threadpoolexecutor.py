import time
from concurrent.futures import ThreadPoolExecutor

def print_num(number):
    time.sleep(2)
    print(number)

a=[1,3,4,5,6]
with ThreadPoolExecutor(max_workers=5) as executor:
    results=executor.map(print_num,a)
for result in results:
    print(result)