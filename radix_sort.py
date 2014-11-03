#!/home/manshu/Softwares/anaconda/bin/python

__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil
from exclusive_scan import preScan

BLOCK_SIZE = 1024
NUM_ELEMENTS = 10000000

@jit(argtypes=[uint32[:], uint32[:], uint32, uint32], target='gpu')
def SplitGPU(in_d, out_d, in_size, bit_shift):
    tx = cuda.threadIdx.x
    index = tx + cuda.blockIdx.x * cuda.blockDim.x
    if index < in_size:
        bit = in_d[index] & (1 << bit_shift)
        if bit > 0:
            bit = 1
        else:
            bit = 0
        out_d[index] = 1 - bit

@jit(argtypes=[uint32[:], uint32[:], uint32, uint32], target='gpu')
def IndexDefineGPU(in_d, rev_bit_d, in_size, last_input):
    tx = cuda.threadIdx.x
    index = tx + cuda.blockIdx.x * cuda.blockDim.x

    total_falses = in_d[in_size - 1] + last_input

    cuda.syncthreads()

    if index < in_size:
        if rev_bit_d[index] == 0:
            in_d[index] = index + 1 - in_d[index] + total_falses

@jit(argtypes=[uint32[:], uint32[:], uint32[:], uint32], target='gpu')
def ScatterElementGPU(in_d, index_d, out_d, in_size):
    tx = cuda.threadIdx.x
    index = tx + cuda.blockIdx.x * cuda.blockDim.x

    if index < in_size and index_d[index] < in_size:
        out_d[index_d[index]] = in_d[index]

def radix_sort(in_d, out_d, out_scan_d, last_inp_element, bit_shift=0):

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (1.0 * threads_per_block[0]))), 1)


    ################ Bit flip ########################
    SplitGPU [number_of_blocks, threads_per_block] (in_d, out_d, NUM_ELEMENTS, bit_shift)

    cuda.synchronize()

    # out_d.copy_to_host(out_h)
    # cuda.synchronize()
    ##################################################
    t1 = time()
    preScan(out_scan_d, out_d, NUM_ELEMENTS)

    cuda.synchronize()
    t2 = time()
    #print "Time = ", t2 - t1

    # out_scan_d.copy_to_host(out_h)
    # cuda.synchronize()
    #
    ###########################################################
    IndexDefineGPU [number_of_blocks, threads_per_block] (out_scan_d, out_d, NUM_ELEMENTS, last_inp_element)

    cuda.synchronize()

    # out_scan_d.copy_to_host(out_h)
    # cuda.synchronize()
    ###########################################################

    ############################################################
    ScatterElementGPU [number_of_blocks, threads_per_block] (in_d, out_scan_d, out_d, NUM_ELEMENTS)

    cuda.synchronize()
    #
    # out_d.copy_to_host(out_h)
    # cuda.synchronize()
    ############################################################

def test_sort():
    in_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)  #4, 7, 2, 6, 3, 5, 1, 0
    out_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)
    for i in range(0, NUM_ELEMENTS):
        in_h[i] = NUM_ELEMENTS - i - 1

    in_d = cuda.to_device(in_h)
    out_d = cuda.device_array(NUM_ELEMENTS, dtype=np.uint32)
    temp_d = cuda.device_array(NUM_ELEMENTS, dtype=np.uint32)

    tkg1 = time()

    for bit_shift in range(0, 32):
        tk1 = time()
        #radix_sort(in_d, out_d, temp_d, in_h[NUM_ELEMENTS - 1], bit_shift)
        preScan(out_d, in_d, NUM_ELEMENTS)
        tk2 = time()
        #print bit_shift, tk2 - tk1
        in_d = out_d
        out_d = temp_d
        temp_d = in_d

    tkg2 = time()

    out_d.copy_to_host(out_h)
    cuda.synchronize()

    # line = ""
    # for i in range(0, NUM_ELEMENTS):
    #     line += " " + str(out_h[i])
    #
    # print line

    in_cpu = [NUM_ELEMENTS - i -1 for i in range(0, NUM_ELEMENTS)]
    tc1 = time()
    in_cpu.sort()
    tc2 = time()

    print "GPU Time = ", tkg2 - tkg1
    print "CPU Time = ", tc2 - tc1


if __name__ == "__main__":
    test_sort()