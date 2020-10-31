import sys,os
import mlb
import plot,test,train,fix
from util import *

def main(cfg):
    mlb.purple('[profiling]')
    import cProfile,pstats
    from pstats import SortKey as sort
    cProfile.runctx('train.train_model(**state.as_kwargs)',globals(),locals(),'profiled')
    p = pstats.Stats('profiled')
    p.strip_dirs()
    p.sort_stats(sort.TIME)
    #p.reverse_order()
    mlb.green('TIME IN FN without children')
    p.sort_stats(sort.TIME)
    p.print_stats(50)
    print('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
    print('tottime: doesnt include subfunctions')
    print('percall: previous column divided by num calls')
    mlb.green('CUMULATIVE')
    p.sort_stats(sort.CUMULATIVE)
    p.print_stats(50)
    print('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)')
    print('tottime: doesnt include subfunctions')
    print('percall: previous column divided by num calls')
    breakpoint()