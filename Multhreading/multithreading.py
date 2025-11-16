import threading
import time

def print_num():
    for i in range(5):
        time.sleep(2)
        print(i)
def print_hello():
    for _ in range(5):
        time.sleep(2)
        print("hello")
time1=time.time()
t1=threading.Thread(target=print_num)
t2=threading.Thread(target=print_hello)
t1.start()
t2.start()
time2=time.time()-time1
print(time2)