import psutil
import os
import time
import gc
import csv

ENABLE_PROFILING = False

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_full_info()
    return mem_info.uss #/ (1024 * 1024) # return in MB

# decorator function
def profile(func):
    if not ENABLE_PROFILING:
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result, None, None
        return wrapper
    
    def wrapper(*args, **kwargs):
        gc.collect()
        gc.disable()
        mem_before = process_memory()
        cpu_before = time.process_time()

        result = func(*args, **kwargs)

        mem_after = process_memory()
        cpu_after = time.process_time()
        gc.enable()
        gc.collect()

        consumed_memory = mem_after - mem_before
        consumed_cpu = cpu_after - cpu_before

        #print("{}:consumed memory: {:,}".format(func.__name__,consumed_memory))
        #print("{}:consumed CPU: {:,}".format(func.__name__,consumed_cpu))

        return result, consumed_memory, consumed_cpu
    return wrapper