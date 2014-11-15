#!/home/manshu/Softwares/anaconda/bin/python
__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil

NUM_ELEMENTS = 12#1000000
MAX_UNIQUE_ITEMS = 7
BLOCK_SIZE = 1024


@jit(argtypes=[int32[:], int32[:], int32], target='gpu')
def histogramGPU(input_d, bins_d, num_elements):
    private_bin = cuda.shared.array(MAX_UNIQUE_ITEMS, uint32)
    tx = cuda.threadIdx.x
    index = cuda.grid(1) #tx + cuda.blockDim.x * cuda.blockIdx.x
    location_x = 0
    for i in range(0, ceil(MAX_UNIQUE_ITEMS / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_UNIQUE_ITEMS:
            private_bin[location_x] = 0

    cuda.syncthreads()

    if index < num_elements and input_d[index] < MAX_UNIQUE_ITEMS:
        cuda.atomic.add(private_bin, input_d[index], 1)
        #cuda.atomic.add(bins_d, input_d[index], 1)

    cuda.syncthreads()

    for i in range(0, ceil(MAX_UNIQUE_ITEMS / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_UNIQUE_ITEMS:
            cuda.atomic.add(bins_d, location_x, private_bin[location_x])

@jit(argtypes=[int32[:], int32, int32], target='gpu')
def pruneGPU(input_d, num_elements, min_sup):
    tx = cuda.threadIdx.x
    index = cuda.grid(1)

    if (index < num_elements):
        if (input_d[index] < min_sup):
            input_d[index] = -1

def test_apriori():
    transactions = [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]
    min_support = 2

    t = [item for transaction in transactions for item in transaction]

    input_h = np.array(t, dtype=np.int32)
    ci_h = np.zeros(MAX_UNIQUE_ITEMS, dtype=np.int32)
    li_h = np.empty(MAX_UNIQUE_ITEMS, dtype=np.int32)

    input_d = cuda.to_device(input_h)
    ci_d = cuda.to_device(ci_h)
    li_d = cuda.device_array(MAX_UNIQUE_ITEMS, dtype=np.int32)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (1.0 * threads_per_block[0]))), 1)#((NUM_ELEMENTS / threads_per_block[0]) + 1, 1)

    t1 = time()
    histogramGPU [number_of_blocks, threads_per_block] (input_d, ci_d, NUM_ELEMENTS)
    cuda.synchronize()

    li_d = ci_d

    number_of_blocks = (int(ceil(MAX_UNIQUE_ITEMS / (1.0 * threads_per_block[0]))), 1)
    pruneGPU [number_of_blocks, threads_per_block] (li_d, MAX_UNIQUE_ITEMS, min_support)
    cuda.synchronize()

    t2 = time()
    li_d.copy_to_host(li_h)


    t3 = time()
    print li_h

if __name__ == "__main__":
    test_apriori()