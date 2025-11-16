from concurrent.futures import ProcessPoolExecutor
import time

def print_num(num):
    time.sleep(5)
    print(num)

number=[1,2,3,4,5,6,7,8,9]
with ProcessPoolExecutor(max_workers=3) as executor:
    results=executor.map(print_num,number)

for result in results:
    print(result)