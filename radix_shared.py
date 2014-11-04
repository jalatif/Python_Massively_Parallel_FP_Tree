#!/home/manshu/Softwares/anaconda/bin/python

__author__ = 'manshu'

from numba import *
from numbapro import vectorize, cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil
from exclusive_scan import preScan

BLOCK_SIZE = 4
SM_SIZE = 2 * BLOCK_SIZE
NUM_ELEMENTS = 1000000
DATA_TYPE = 32


@jit(argtypes=[uint32[:], uint32[:], uint32], target='gpu')
def RadixGPU(in_d, out_d, in_size):
    private_shared_in = cuda.shared.array(SM_SIZE, uint32)
    private_split = cuda.shared.array(SM_SIZE, uint32)
    private_scan = cuda.shared.array(SM_SIZE, uint32)

    start = 2 * cuda.blockDim.x * cuda.blockIdx.x
    tx = cuda.threadIdx.x
    index = tx + start

    ############### Put 2 values per each thread into shared memory ##############
    if index < in_size:
        private_shared_in[tx] = in_d[index]
    else:
        private_shared_in[tx] = 2 ** (DATA_TYPE - 1) #0xffffffff

    if (index + BLOCK_SIZE) < in_size:
        private_shared_in[tx + BLOCK_SIZE] = in_d[index + BLOCK_SIZE]
    else:
        private_shared_in[tx + BLOCK_SIZE] = 2 ** (DATA_TYPE - 1) #0xffffffff

    cuda.syncthreads()

    total_falses = 0.0
    t = 0
    f = 0
    bit = 0
    d = 1
    for bit_shift in range(0, DATA_TYPE):
        bit = private_shared_in[tx] & (1 << bit_shift)
        if bit > 0:
            bit = 1
        private_split[tx] = 1 - bit
        private_scan[tx] = 1 - bit

        bit = private_shared_in[tx + BLOCK_SIZE] & (1 << bit_shift)
        if bit > 0:
            bit = 1
        private_split[tx + BLOCK_SIZE] = 1 - bit
        private_scan[tx + BLOCK_SIZE] = 1 - bit

        cuda.syncthreads()

        ########################### Do the first scan ##############################
        d = 1
        while d <= BLOCK_SIZE:
            tk = 2 * d * (tx + 1) - 1
            if tk < (2 * BLOCK_SIZE):
                private_scan[tk] += private_scan[tk - d]
            d *= 2
            cuda.syncthreads()

        ############################ Do the second scan #############################

        d = BLOCK_SIZE / 2
        while d > 0:
            tk = 2 * d * (tx + 1) - 1
            if (tk + d) < (2 * BLOCK_SIZE):
                private_scan[tk + d] += private_scan[tk]
            d /= 2
            cuda.syncthreads()

        #############################################################################
        # temp_index = tx + 1
        # if index < in_size:
        #     private_split_ex[temp_index] = private_split[tx]
        # if (index + BLOCK_SIZE) < in_size and (tx + BLOCK_SIZE) != (2 * BLOCK_SIZE - 1):
        #     private_split_ex[temp_index + BLOCK_SIZE] = private_split[tx + BLOCK_SIZE]
        # total_falses = private_split[2 * BLOCK_SIZE - 1]
        # private_split_ex[start] = 0.0

        total_falses = private_scan[SM_SIZE - 1]
        t = total_falses
        f = 0
        if tx != 0:
            t = tx - private_scan[tx - 1] + total_falses
            f = private_scan[tx - 1]
        if private_split[tx] == 1:
            private_split[tx] = f
        else:
            private_split[tx] = t

        t = (tx + BLOCK_SIZE) - private_scan[tx + BLOCK_SIZE - 1] + total_falses
        f = private_scan[tx + BLOCK_SIZE - 1]
        if private_split[tx + BLOCK_SIZE] == 1:
            private_split[tx + BLOCK_SIZE] = f
        else:
            private_split[tx + BLOCK_SIZE] = t

        cuda.syncthreads()

        private_scan[private_split[tx]] = private_shared_in[tx]
        private_scan[private_split[tx + BLOCK_SIZE]] = private_shared_in[tx + BLOCK_SIZE]

        cuda.syncthreads()

        private_shared_in[tx] = private_scan[tx]
        private_shared_in[tx + BLOCK_SIZE] = private_scan[tx + BLOCK_SIZE]

        cuda.syncthreads()

    if index < in_size:
        out_d[index] = private_shared_in[tx]

    if (index + BLOCK_SIZE) < in_size:
        out_d[index + BLOCK_SIZE] = private_shared_in[tx + BLOCK_SIZE]


def test_sort():
    in_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)  #4, 7, 2, 6, 3, 5, 1, 0
    #in_h = np.array([4, 7, 2, 6, 3, 5, 1, 0], dtype=np.uint32)
    out_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)
    for i in range(0, NUM_ELEMENTS):
        in_h[i] = NUM_ELEMENTS - i - 1

    in_d = cuda.to_device(in_h)
    out_d = cuda.device_array(NUM_ELEMENTS, dtype=np.uint32)
    #temp_d = cuda.device_array(NUM_ELEMENTS, dtype=np.uint32)

    tkg1 = time()

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (2 * 1.0 * threads_per_block[0]))), 1)

    RadixGPU [number_of_blocks, threads_per_block] (in_d, out_d, NUM_ELEMENTS)

    tkg2 = time()

    out_d.copy_to_host(out_h)
    cuda.synchronize()
    #
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