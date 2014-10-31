#!/home/manshu/Softwares/anaconda/bin/python

__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil

BLOCK_SIZE = 512
SM_SIZE = 1024
NUM_ELEMENTS = 1024

@jit(argtypes=[float32[:], float32[:], float32[:], uint32], target='gpu')
def exclusiveScanGPU(aux_d, out_d, in_d, size):
    private_shared_in = cuda.shared.array(SM_SIZE, float32)
    start = 2 * cuda.blockDim.x * cuda.blockIdx.x
    tx = cuda.threadIdx.x
    index = tx + start
    ############### Put 2 values per each thread into shared memory ##############
    if index < size:
        private_shared_in[tx] = in_d[index]
    else:
        private_shared_in[tx] = 0.0

    if (index + BLOCK_SIZE) < size:
        private_shared_in[tx + BLOCK_SIZE] = in_d[index + BLOCK_SIZE]
    else:
        private_shared_in[tx + BLOCK_SIZE] = 0.0

    cuda.syncthreads()
    ########################### Do the first scan ##############################
    d = 1
    while d <= BLOCK_SIZE:
        tk = 2 * d * (tx + 1) - 1
        if tk < (2 * BLOCK_SIZE):
            private_shared_in[tk] += private_shared_in[tk - d]
        d *= 2
        cuda.syncthreads()

    ############################ Do the second scan #############################

    d = BLOCK_SIZE / 2
    while d > 0:
        tk = 2 * d * (tx + 1) - 1
        if (tk + d) < (2 * BLOCK_SIZE):
            private_shared_in[tk + d] += private_shared_in[tk]
        d /= 2
        cuda.syncthreads()

    #############################################################################

    index += 1

    if index < size:
        out_d[index] = private_shared_in[tx]

    if (index + BLOCK_SIZE) < size and (tx + BLOCK_SIZE) != (2 * BLOCK_SIZE - 1):
        out_d[index + BLOCK_SIZE] = private_shared_in[tx + BLOCK_SIZE]

    cuda.syncthreads()

    aux_d[cuda.blockIdx.x] = private_shared_in[2 * BLOCK_SIZE - 1]
    out_d[start] = 0.0


@jit(argtypes=[float32[:], float32[:], uint32], target='gpu')
def exclusiveCombineGPU(out_d, aux_in, size):
    tx = cuda.threadIdx.x
    start = 2 * cuda.blockIdx.x * cuda.blockDim.x
    index = start + tx

    if cuda.blockIdx.x != 0 and index < size:
        out_d[index] += aux_in[cuda.blockIdx.x]

    if cuda.blockIdx.x != 0 and (index + BLOCK_SIZE) < size:
        out_d[index + BLOCK_SIZE] += aux_in[cuda.blockIdx.x]


def preScan(out_d, in_d, in_size):
    threads_per_block = (BLOCK_SIZE, 1)
    nBlocks = int(ceil(in_size / (2 * 1.0 * BLOCK_SIZE)))
    number_of_blocks = (nBlocks, 1)

    aux_h = np.empty(nBlocks, dtype=np.float32)
    aux_oh = np.empty(nBlocks, dtype=np.float32)

    aux_d = cuda.to_device(aux_h)
    aux_od = cuda.to_device(aux_oh)

    exclusiveScanGPU [number_of_blocks, threads_per_block] (aux_d, out_d, in_d, in_size)

    aux_d.copy_to_host(aux_h)

    out_h = np.empty(in_size, dtype=np.float32)

    out_d.copy_to_host(out_h)

    if nBlocks > 1:
        preScan(aux_od, aux_d, nBlocks)
    else:
        aux_od = aux_d

    exclusiveCombineGPU [number_of_blocks, threads_per_block] (out_d, aux_od, in_size)

    out_d.copy_to_host(out_h)

def main():


    in_h = np.empty(NUM_ELEMENTS, dtype=np.float32)
    out_h = np.zeros(NUM_ELEMENTS, dtype=np.float32)

    for i in range(0, NUM_ELEMENTS):
        in_h[i] = 1#srandint(0, 100)

    in_d = cuda.to_device(in_h)
    out_d = cuda.to_device(out_h)

    cuda.synchronize()

    preScan(out_d, in_d, NUM_ELEMENTS)

    cuda.synchronize()

    out_d.copy_to_host(out_h)

    cuda.synchronize()

    for i in range(0, NUM_ELEMENTS):
        print out_h[i]

main()