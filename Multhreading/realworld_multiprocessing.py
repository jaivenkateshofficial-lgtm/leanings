import multiprocessing
import sys
import time
import math

sys.set_int_max_str_digits(1000000)
sys.setrecursionlimit(100000)

def compute_fact(num):
    time.sleep(2)
    return math.factorial(num)
    # if num == 1 or num == 0:
    #     return 1
    # else:
    #     return num * compute_fact(num - 1)


if __name__ == "__main__":  
    numbers = [50000, 3000,4000,7000,8000,200]
    start = time.time()

    with multiprocessing.Pool() as pool:  
        results = pool.map(compute_fact, numbers) 

    end_time = time.time()
    print(f' time with multi{end_time - start:.2f} seconds')

    start=time.time()
    a=[]
    for num in numbers:
        a.append(compute_fact(num))
    end=time.time()
    print(f'time without multi{end-start:.6f}')