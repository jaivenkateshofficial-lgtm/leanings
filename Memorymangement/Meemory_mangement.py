# Reffremce count is the python  primary method to allocate and deallocate memory
# when reffrence count is zero python deallocates memoery
import sys
import gc
a=[]
b=a
print(sys.getrefcount(a))
del b
print(sys.getrefcount(a))

# Garbage collection
"""
Pyhon includes cyclic garbage collector to handle reffrence cycle ,Reffrence cycle occur when
 refrrence each other and prevents the reffrence count to zero """
gc.enable()
z=0
gc.collect()
print(z)
print(gc.get_stats())
print(gc.garbage)
"""
1.use most loacl variable
2.avoid cycle reffrence
3.explisitly delete object
4.use genrator which make it keep one object at a time
"""