#!/home/manshu/Softwares/anaconda/bin/python

__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil
import exclusive_scan

BLOCK_SIZE = 4
NUM_ELEMENTS = 7

@jit(argtypes=[uint32[:], uint32[:], uint32], target='gpu')
def SplitGPU(in_d, out_d, in_size):
    tx = cuda.threadIdx.x
    index = tx + cuda.blockIdx.x * cuda.blockDim.x
    if index < in_size:
        bit = in_d[index] & 0x00000001
        out_d[index] = 1 - bit

def main():

    in_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)
    out_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)
    out_scan_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)

    for i in range(0, NUM_ELEMENTS):
        in_h[i] = NUM_ELEMENTS - i - 1

    in_d = cuda.to_device(in_h)
    out_d = cuda.to_device(out_h)
    out_scan_d = cuda.to_device(out_h)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (1.0 * threads_per_block[0]))), 1)


    ################ Bit flip ########################
    SplitGPU [number_of_blocks, threads_per_block] (in_d, out_d, NUM_ELEMENTS)

    cuda.synchronize()

    out_d.copy_to_host(out_h)

    cuda.synchronize()
    ##################################################

    line = ""
    for i in range(0, NUM_ELEMENTS):
        line += " " + str(out_h[i])

    print line

    ##################################################

    ##################################################
    exclusive_scan.preScan(out_scan_d, out_d, NUM_ELEMENTS)

    cuda.synchronize()

    out_scan_d.copy_to_host(out_scan_h)
    total_falses = out_scan_h[-1] + out_h[-1]
    print "Total Falses = ", total_falses
    ###################################################
    line = ""
    for i in range(0, NUM_ELEMENTS):
        line += " " + str(out_scan_h[i])

    print line

main()