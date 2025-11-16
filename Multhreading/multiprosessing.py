import multiprocessing
import multiprocessing.process
import time

def print_number():
    for i in range(5):
        time.sleep(2)
        print(i)
def print_hello():
    for _ in range(5):
        time.sleep(5)
        print("hello")
time1=time.time()
if __name__ == "__main__":
    t1=multiprocessing.Process(target=print_number)
    t2=multiprocessing.Process(target=print_hello)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    time2=time.time()-time1
    print(time2)