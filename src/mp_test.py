#!/usr/bin/env python
import multiprocessing as mp


def ff(a,b):
    return a+b

def f(x):
    return ff(x[0],x[1])

def fff(a):
    return a*2

p = mp.Pool()
res=p.map(f, [(1,2), (3,4)])
print(res)
print(mp.cpu_count())


res_list=[p.apply_async(f,((1,i),)) for i in range(10)]
a=[res.get() for res in res_list]
print(a)
