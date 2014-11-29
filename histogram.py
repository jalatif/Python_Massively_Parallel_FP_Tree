#!/home/manshu/Softwares/anaconda/bin/python

__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil

NUM_ELEMENTS = 1000000
BIN_SIZE = 12000
BLOCK_SIZE = 1024


#@cuda.jit('void(uint32[:], uint32[:], uint32, uint32)')
#@cuda.autojit

@jit(argtypes=[uint32[:], uint32[:], uint32], target='gpu')
def histogramGPU(input_d, bins_d, num_elements):
    private_bin = cuda.shared.array(BIN_SIZE, uint32)
    tx = cuda.threadIdx.x
    index = cuda.grid(1) #tx + cuda.blockDim.x * cuda.blockIdx.x
    location_x = 0
    for i in range(0, ceil(BIN_SIZE / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < BIN_SIZE:
            private_bin[location_x] = 0

    cuda.syncthreads()

    if index < num_elements and input_d[index] < BIN_SIZE:
        cuda.atomic.add(private_bin, input_d[index], 1)
        #cuda.atomic.add(bins_d, input_d[index], 1)

    cuda.syncthreads()

    for i in range(0, ceil(BIN_SIZE / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < BIN_SIZE:
            cuda.atomic.add(bins_d, location_x, private_bin[location_x])


def myprint(string):
    print string

def makeHist(input_h):
    bins_h = np.zeros(BIN_SIZE, dtype=np.uint32)

    for input in input_h:
        bins_h[input] += 1

    return bins_h

def test_histogram():
    #Allocate host memory
    input_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)
    bins_h = np.zeros(BIN_SIZE, dtype=np.uint32)
    myprint("Bin Size = " + str(bins_h.size))
    ## Initialize host memory
    for i in range(0, NUM_ELEMENTS):
        input_h[i] = randint(0, BIN_SIZE - 1)

    ## Allocate and initialize GPU/device memory
    input_d = cuda.to_device(input_h)
    bins_d = cuda.to_device(bins_h)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (1.0 * threads_per_block[0]))), 1)#((NUM_ELEMENTS / threads_per_block[0]) + 1, 1)

    t1 = time()
    histogramGPU [number_of_blocks, threads_per_block] (input_d, bins_d, NUM_ELEMENTS)
    cuda.synchronize()
    t2 = time()
    bins_d.copy_to_host(bins_h)

    t3 = time()
    bins_cpu = makeHist(input_h)
    t4 = time()

    # for i in range(0, BIN_SIZE):
    #     print i, bins_h[i], bins_cpu[i]

    print "GPU time = ", t2 - t1
    print "CPU TIME = ", t4 - t3

    match = 1
    for i in range(0, BIN_SIZE):
        if bins_h[i] != bins_cpu[i]:
            match = -1
            break
    if match == 1:
        print "Test Passed"
    else:
        print "Test Failed"

test_histogram() # Run the program
