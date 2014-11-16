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
NUM_ELEMENTS = 10000000#1024 * 1024 * 128
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

#
# @jit(argtypes=[uint32[:], uint32, uint32], target='gpu')
# def bitonicSort(in_d, in_size, stride):
#     tx = cuda.threadIdx.x
#     start = stride * cuda.blockDim.x * cuda.blockIdx.x
#     pstart = start
#     index = start + tx
#
#     for i in range(0, stride / 2):
#         k = stride - 2 * i
#         pstart = start + i * BLOCK_SIZE
#
#         v1 = in_d[pstart + tx]
#         v2 = in_d[pstart + k * BLOCK_SIZE - 1 - tx]
#
#         if v2 > v1:
#             min1 = v1
#             max1 = v2
#         else:
#             min1 = v2
#             max1 = v1
#
#         in_d[pstart + tx] = min1
#         in_d[pstart + k * BLOCK_SIZE - 1 - tx] = max1
#
#     cuda.syncthreads()
#     stride /= 2
#
#     j = 2
#     while stride >= 0:
#         for i in range(0, j * stride / 2):
#             pstart = start + 2 * i * BLOCK_SIZE
#
#             v1 = in_d[pstart + tx]
#             v2 = in_d[pstart + tx + stride / 2 * BLOCK_SIZE]
#
#             if v2 > v1:
#                 min1 = v1
#                 max1 = v2
#             else:
#                 min1 = v2
#                 max1 = v1
#
#             in_d[pstart + tx] = min1
#             in_d[pstart + tx + stride / 2 * BLOCK_SIZE] = max1
#
#         cuda.syncthreads()
#         if stride == 1:
#             break
#         stride /= 2
#         j *= 2

# @jit(argtypes=[uint32[:], uint32], target='gpu')
# def is_sorted(in_d, out_bool):
#     tx = cuda.threadIdx.x
#     index = tx + cuda.blockDim.x * cuda.blockIdx.x
#

def test_sort():
    in_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)  #4, 7, 2, 6, 3, 5, 1, 0
    #in_h = np.array([4, 7, 2, 6, 3, 5, 1, 0], dtype=np.uint32)
    out_h = np.empty(NUM_ELEMENTS, dtype=np.uint32)
    for i in range(0, NUM_ELEMENTS):
        in_h[i] = randint(0, 100)#NUM_ELEMENTS - i - 1
    #in_h = np.array([6, 44, 71, 79, 94, 92, 12, 56, 47, 17, 81, 98, 84,  9, 85, 99], dtype=np.uint32)
    #in_h = np.array([85, 37, 50, 73, 51, 46, 62, 84, 65, 99, 76, 59, 73, 16, 27, 4, 75, 81, 80, 33, 73, 11, 29, 24, 81, 49, 27, 71, 74, 64, 60, 91], dtype=np.uint32)
    print in_h

    in_d = cuda.to_device(in_h)
    out_d = cuda.device_array(NUM_ELEMENTS, dtype=np.uint32)

    tkg1 = time()

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (2 * 1.0 * threads_per_block[0]))), 1)

    RadixGPU [number_of_blocks, threads_per_block] (in_d, out_d, NUM_ELEMENTS)
    out_d.copy_to_host(out_h)
    #print "Rad = ", list(out_h)

    stride = 4
    # while stride < NUM_ELEMENTS:
    #     number_of_blocks = (int(ceil(NUM_ELEMENTS / (stride * 1.0 * threads_per_block[0]))), 1)
    #     bitonicSort [number_of_blocks, threads_per_block] (out_d, NUM_ELEMENTS, stride)
    #     stride *= 2
    #     # number_of_blocks = (int(ceil(NUM_ELEMENTS / (2 * 1.0 * threads_per_block[0]))), 1)
    #     # RadixGPU [number_of_blocks, threads_per_block] (out_d, in_d, NUM_ELEMENTS)
    #     # out_d = in_d
    #     out_d.copy_to_host(out_h)
    #     print "Str = ", list(out_h)
    #     break
    # # stride /= 2
    # while stride >= 4:
    #     number_of_blocks = (int(ceil(NUM_ELEMENTS / (stride * 1.0 * threads_per_block[0]))), 1)
    #     bitonicSort [number_of_blocks, threads_per_block] (out_d, NUM_ELEMENTS, stride)
    #     stride /= 2
    #     cuda.synchronize()
    #
    #     number_of_blocks = (int(ceil(NUM_ELEMENTS / (2 * 1.0 * threads_per_block[0]))), 1)
    #     RadixGPU [number_of_blocks, threads_per_block] (out_d, in_d, NUM_ELEMENTS)
    #     out_d = in_d
        #
        # out_d.copy_to_host(out_h)
        # cuda.synchronize()
        #
        # line = ""
        # for i in range(0, NUM_ELEMENTS):
        #     line += " " + str(out_h[i])
        #
        # print line

    tkg2 = time()

    out_d.copy_to_host(out_h)
    cuda.synchronize()
    #print "GPU = ", list(out_h)
    # line = ""
    # for i in range(0, NUM_ELEMENTS):
    #     line += " " + str(out_h[i])
    #
    # print line

    in_cpu = list(in_h)#[NUM_ELEMENTS - i -1 for i in range(0, NUM_ELEMENTS)]
    tc1 = time()
    in_cpu.sort()
    #print "CPU = ", in_cpu
    tc2 = time()

    print "GPU Time = ", tkg2 - tkg1
    print "CPU Time = ", tc2 - tc1
    print len(in_cpu)


if __name__ == "__main__":
    test_sort()