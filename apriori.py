#!/home/manshu/Softwares/anaconda/bin/python
__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil
from pprint import pprint
import sys

NUM_ELEMENTS = 12#1000000
MAX_UNIQUE_ITEMS = 7
BLOCK_SIZE = 2
MAX_ITEM_PER_SM = 4
MAX_TRANSACTIONS_PER_SM = 10
MAX_ITEMS_PER_TRANSACTIONS = 10
MAX_TRANSACTIONS = 4
SM_SHAPE = MAX_TRANSACTIONS_PER_SM * MAX_ITEMS_PER_TRANSACTIONS

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
            input_d[index] = 0

@jit(argtypes=[int32[:], int32[:], int32], target='gpu')
def selfJoinGPU(input_d, output_d, num_elements):
    tx = cuda.threadIdx.x
    #index = tx + cuda.blockIdx.x * cuda.blockDim.x
    start = cuda.blockIdx.x * MAX_ITEM_PER_SM

    sm1 = cuda.shared.array(MAX_ITEM_PER_SM, int32)
    sm2 = cuda.shared.array(MAX_ITEM_PER_SM, int32)

    for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < num_elements:
            sm1[location_x] = input_d[start + location_x]

    cuda.syncthreads()


    for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
        loop_tx = tx + i * BLOCK_SIZE
        for j in range(loop_tx + 1, MAX_ITEM_PER_SM):
            if (sm1[loop_tx] / 10) == (sm1[j] / 10):
                output_d[(start + loop_tx) * num_elements + (start + j)] = 1


    for smid in range(cuda.blockIdx.x + 1, ceil(num_elements / (1.0 * MAX_ITEM_PER_SM))):
        for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
            location_x = tx + i * BLOCK_SIZE
            if location_x < num_elements:
                sm2[location_x] = input_d[smid * MAX_ITEM_PER_SM + start + location_x]

        cuda.syncthreads()

        for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
            loop_tx = tx + i * BLOCK_SIZE
            for j in range(0, MAX_ITEM_PER_SM):
                if (sm1[loop_tx] / 10) == (sm2[j] / 10):
                    output_d[(start + loop_tx) * num_elements + smid * MAX_ITEM_PER_SM + j] = 1

    #cuda.syncthreads()

@jit(argtypes=[int32[:], int32[:], int32, int32[:], int32[:,:], int32], target='gpu')
def findFrequencyGPU(d_transactions, d_offsets, num_transactions, dkeyIndex, dMask, num_patterns):

    Ts = cuda.shared.array(SM_SHAPE, int32)
    transaction_start_index = cuda.blockDim.x * cuda.blockIdx.x

    while transaction_start_index < num_transactions:
        index = cuda.threadIdx.x
        transaction_end_index = transaction_start_index + cuda.blockDim.x

        cuda.syncthreads()

        #Clear SM

        for i in range(0, MAX_TRANSACTIONS_PER_SM):
            while index < MAX_ITEMS_PER_TRANSACTIONS:
                Ts[i * MAX_ITEMS_PER_TRANSACTIONS + index] = 0
                index += cuda.blockDim.x

            cuda.syncthreads()

        for i in range(transaction_start_index, transaction_end_index):
            if i >= num_transactions:
                break
            start_offset = d_offsets[i]
            end_offset = d_offsets[i + 1]
            index1 = start_offset + cuda.threadIdx.x

            cuda.syncthreads()

            #threads collaborate to get the ith transaction

            while index1 < end_offset:
                Ts[(i - transaction_start_index) * MAX_ITEMS_PER_TRANSACTIONS + (index1 - start_offset)] = d_transactions[index1]
                index1 += cuda.blockDim.x

            cuda.syncthreads()

        if cuda.threadIdx.x < MAX_TRANSACTIONS_PER_SM:
            for j in range(0, MAX_ITEMS_PER_TRANSACTIONS):
                Ts[cuda.threadIdx.x * MAX_ITEMS_PER_TRANSACTIONS + j] += 1

        cuda.syncthreads()
        for i in range(transaction_start_index, transaction_end_index):
            if i >= num_transactions:
                break

            start_offset = d_offsets[i]
            end_offset = d_offsets[i + 1]
            index1 = start_offset + cuda.threadIdx.x

            cuda.syncthreads()

            #threads collaborate to get the ith transaction

            while index1 < end_offset:
                d_transactions[index1] = Ts[(i - transaction_start_index) * MAX_ITEMS_PER_TRANSACTIONS + (index1 - start_offset)]
                index1 += cuda.blockDim.x

            cuda.syncthreads()

        transaction_start_index += cuda.blockDim.x * cuda.gridDim.x

def selfJoinCPU(input_cpu):
    output_cpu = []
    for i in range(0, len(input_cpu)):
        for j in range(i, len(input_cpu)):
            if (input_cpu[i] / 10) == (input_cpu[j] / 10):
                output_cpu.append((input_cpu[i], input_cpu[j]))
    return output_cpu

def readFile(file_name):
    fp = open(file_name, 'r')

    transactions = np.empty(MAX_ITEMS_PER_TRANSACTIONS * MAX_TRANSACTIONS, dtype=np.int32)
    offsets = np.empty(MAX_TRANSACTIONS + 1, dtype=np.int32)
    offsets[0] = 0

    trans_id = 0
    lines = 0

    for line in fp:
        if lines >= MAX_TRANSACTIONS: break
        line = line.strip()
        words = line.split(' ')
        for word in words:
            transactions[trans_id] = int(word)
            trans_id += 1
        offsets[lines + 1] = offsets[lines] + len(words)
        lines += 1

    return offsets, transactions, lines, trans_id

def test_apriori():

    offsets, transactions, num_transactions, num_elements = readFile("dummy.txt")
    print offsets[:num_transactions]
    print transactions[:num_transactions]
    print num_transactions
    print num_elements
    min_support = 2

    t = [item for item in transactions.tolist()]


    input_h = np.array(t, dtype=np.int32)
    ci_h = np.zeros(MAX_UNIQUE_ITEMS, dtype=np.int32)
    li_h = np.empty(MAX_UNIQUE_ITEMS, dtype=np.int32)

    input_d = cuda.to_device(input_h)
    ci_d = cuda.to_device(ci_h)
    li_d = cuda.device_array(MAX_UNIQUE_ITEMS, dtype=np.int32)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (1.0 * threads_per_block[0]))), 1)#((NUM_ELEMENTS / threads_per_block[0]) + 1, 1)

    histogramGPU [number_of_blocks, threads_per_block] (input_d, ci_d, NUM_ELEMENTS)
    cuda.synchronize()

    number_of_blocks = (int(ceil(MAX_UNIQUE_ITEMS / (1.0 * threads_per_block[0]))), 1)
    pruneGPU [number_of_blocks, threads_per_block] (ci_d, MAX_UNIQUE_ITEMS, min_support)
    cuda.synchronize()

    ci_d.copy_to_host(ci_h)
    print ci_h

    k = 0
    for j in range(0, len(ci_h)):
        if ci_h[j] != 0:
            li_h[k] = j
            k += 1

    print li_h, k

    # k = 10000

    ci_h = np.zeros(k ** 2, dtype=np.int32)
    ci_d = cuda.to_device(ci_h)

    #li_h = np.array(sorted([randint(10, 99) for i in range(0, k)]), dtype=np.int32)

    t1 = time()
    li_d = cuda.to_device(li_h)
    number_of_blocks = (int(ceil(k / MAX_ITEM_PER_SM)), 1)
    selfJoinGPU [number_of_blocks, threads_per_block](li_d, ci_d, k)

    ci_d.copy_to_host(ci_h)
    t2 = time()

    print t2 - t1

    d_offsets = cuda.to_device(offsets)
    d_transactions = cuda.to_device(transactions)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(num_transactions / (1.0 * threads_per_block[0])) + 1, 1)

    findFrequencyGPU [number_of_blocks, threads_per_block] (d_transactions, d_offsets, num_transactions, li_d, ci_d, k)

    d_transactions.copy_to_host(transactions)

    print transactions[:num_elements]

    li_cpu = list(li_h)

    t3 = time()
    #selfJoinCPU(li_cpu)
    t4 = time()
    print t4 - t3
    ci_h = ci_h.reshape(k, k)

    print(ci_h.tolist())

if __name__ == "__main__":
    test_apriori()