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
MAX_UNIQUE_ITEMS = 6
BLOCK_SIZE = 4
SM_SIZE = 2 * BLOCK_SIZE
MAX_ITEM_PER_SM = 4
MAX_TRANSACTIONS_PER_SM = 2
MAX_ITEMS_PER_TRANSACTIONS = 4
MAX_TRANSACTIONS = 4
SM_SHAPE = (MAX_TRANSACTIONS_PER_SM, MAX_ITEMS_PER_TRANSACTIONS)

@jit(argtypes=[uint32[:], uint32[:], uint32[:], uint32], target='gpu')
def exclusiveScanGPU(aux_d, out_d, in_d, size):
    private_shared_in = cuda.shared.array(SM_SIZE, uint32)
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


@jit(argtypes=[uint32[:], uint32[:], uint32], target='gpu')
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

    aux_d = cuda.device_array(nBlocks, dtype=np.uint32)
    aux_od = cuda.device_array(nBlocks, dtype=np.uint32)

    exclusiveScanGPU [number_of_blocks, threads_per_block] (aux_d, out_d, in_d, in_size)

    if nBlocks > 1:
        preScan(aux_od, aux_d, nBlocks)
    else:
        aux_od = aux_d

    exclusiveCombineGPU [number_of_blocks, threads_per_block] (out_d, aux_od, in_size)

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

    if index < num_elements:
        if input_d[index] < min_sup:
            input_d[index] = 0

@jit(argtypes=[int32[:], int32, int32], target='gpu')
def pruneMultipleGPU(input_d, num_patterns, min_sup):

    index_x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    index_y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    data_index = index_y * num_patterns + index_x

    if data_index < num_patterns * num_patterns:
        if input_d[data_index] < min_sup:
            input_d[data_index] = 0

@jit(argtypes=[int32[:], int32[:], int32], target='gpu')
def selfJoinGPU(input_d, output_d, num_elements):
    tx = cuda.threadIdx.x
    #index = tx + cuda.blockIdx.x * cuda.blockDim.x
    start = cuda.blockIdx.x * MAX_ITEM_PER_SM

    sm1 = cuda.shared.array(MAX_ITEM_PER_SM, int32)
    sm2 = cuda.shared.array(MAX_ITEM_PER_SM, int32)

    for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_ITEM_PER_SM:
            sm1[location_x] = input_d[start + location_x]

    cuda.syncthreads()


    for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
        loop_tx = tx + i * BLOCK_SIZE
        for j in range(loop_tx + 1, MAX_ITEM_PER_SM):
            if (sm1[loop_tx] / 10) == (sm1[j] / 10):
                output_d[(start + loop_tx) * num_elements + (start + j)] = 0
            else:
                output_d[(start + loop_tx) * num_elements + (start + j)] = -1


    for smid in range(cuda.blockIdx.x + 1, ceil(num_elements / (1.0 * MAX_ITEM_PER_SM))):
        for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
            location_x = tx + i * BLOCK_SIZE
            if location_x < MAX_ITEM_PER_SM:
                sm2[location_x] = input_d[smid * MAX_ITEM_PER_SM + start + location_x]

        cuda.syncthreads()

        for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
            loop_tx = tx + i * BLOCK_SIZE
            for j in range(0, MAX_ITEM_PER_SM):
                if (sm1[loop_tx] / 10) == (sm2[j] / 10):
                    output_d[(start + loop_tx) * num_elements + smid * MAX_ITEM_PER_SM + j] = 0
                else:
                    output_d[(start + loop_tx) * num_elements + smid * MAX_ITEM_PER_SM + j] = -1

    #cuda.syncthreads()

@jit(argtypes=[int32[:], int32[:], int32, int32, int32[:], int32[:], int32], target='gpu')
def findFrequencyGPU(d_transactions, d_offsets, num_transactions, num_elements, dkeyIndex, dMask, num_patterns):
    Ts = cuda.shared.array(SM_SHAPE, int32)
    tx = cuda.threadIdx.x

    index = tx + cuda.blockDim.x * cuda.blockIdx.x
    trans_index = cuda.blockIdx.x * MAX_TRANSACTIONS_PER_SM

    for i in range(0, MAX_TRANSACTIONS_PER_SM):
        if tx < MAX_ITEMS_PER_TRANSACTIONS:
            Ts[i, tx] = -1

    cuda.syncthreads()

    for i in range(0, MAX_TRANSACTIONS_PER_SM):
        item_ends = num_elements
        if (trans_index + i + 1) == num_transactions:
            item_ends = num_elements
        elif (trans_index + i + 1) < num_transactions:
            item_ends = d_offsets[trans_index + i + 1]
        else:
            continue
        if (tx + d_offsets[trans_index + i]) < item_ends and tx < MAX_ITEMS_PER_TRANSACTIONS:
            Ts[i, tx] = d_transactions[d_offsets[trans_index + i] + tx]
            #d_transactions[d_offsets[trans_index + i] + tx] += 1

    cuda.syncthreads()

    for mask_id in range(0, int(ceil(num_patterns / 1.0 * cuda.blockDim.x))):
        loop_tx = cuda.threadIdx.x + mask_id * cuda.blockDim.x

        for last_seen in range(0, num_patterns):
            if dMask[loop_tx * num_patterns + last_seen] < 0:
                last_seen += 1
                continue
            item1 = dkeyIndex[loop_tx]
            item2 = dkeyIndex[last_seen]
            for tid in range(0, MAX_TRANSACTIONS_PER_SM):
                flag1 = False
                flag2 = False
                for titem in range(0, MAX_ITEMS_PER_TRANSACTIONS):
                    if Ts[tid, titem] == item1: flag1 = True
                    elif Ts[tid, titem] == item2: flag2 = True

                present_flag = flag1 and flag2
                if present_flag:
                    cuda.atomic.add(dMask, loop_tx * num_patterns + last_seen, 1)

@jit(argtypes=[int32[:], int32[:], int32], target='gpu')
def combinationsAvailable(input_d, output_d, k):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    index_x = tx + cuda.blockIdx.x * cuda.blockDim.x
    index_y = ty + cuda.blockIdx.y * cuda.blockDim.y

    if index_x < k and index_y < k and input_d[index_y * k + index_x] > 0:
        cuda.atomic.add(output_d, index_y, 1)

@jit(argtypes=[int32[:], int32[:], int32[:], int32], target='gpu')
def convert2Sparse(input_d, offset_d, output_d, k):
    tx = cuda.threadIdx.x
    index = tx + cuda.blockIdx.x * cuda.blockDim.x

    if index < (k - 1):
        col_start = offset_d[index]
        #col_end = offset_d[index + 1]
        for i in range(0, k):
            support_value = input_d[index * k + i]
            if support_value > 0:
                output_d[col_start] = index
                output_d[col_start + k] = i
                output_d[col_start + 2 * k] = support_value
                col_start += 1


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

    #k = 10000

    ci_h = np.array([-1 for i in range(0, k ** 2)], dtype=np.int32)
    ci_d = cuda.to_device(ci_h)

    #li_h = np.array(sorted([randint(10, 99) for i in range(0, k)]), dtype=np.int32)

    t1 = time()
    li_d = cuda.to_device(li_h)
    number_of_blocks = (int(ceil(k / MAX_ITEM_PER_SM)), 1)
    selfJoinGPU [number_of_blocks, threads_per_block](li_d, ci_d, k)

    li_d.copy_to_host(li_h)
    ci_d.copy_to_host(ci_h)
    t2 = time()


    #ci_h = ci_h.reshape(k, k)

    print "Initial Mask = ", ci_h

    print t2 - t1

    d_offsets = cuda.to_device(offsets)
    d_transactions = cuda.to_device(transactions)

    threads_per_block = (BLOCK_SIZE, 1)
    #number_of_blocks = (1, 1) #(int(num_transactions / (1.0 * threads_per_block[0])) + 1, 1)
    number_of_blocks = (int(ceil(num_transactions / (1.0 * MAX_TRANSACTIONS_PER_SM))), 1)

    print "Num transactions = ", num_transactions
    print "Num patterns = ", k
    print "index = ", li_h
    findFrequencyGPU [number_of_blocks, threads_per_block] (d_transactions, d_offsets, num_transactions, num_elements, li_d, ci_d, k)
    cuda.synchronize()
    ci_d.copy_to_host(ci_h)
    print "Final Mask = ", ci_h.reshape(4, 4)
    d_transactions.copy_to_host(transactions)

    print transactions[:num_elements]

    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    number_of_blocks = ((int(ceil(k / (1.0 * threads_per_block[0])))), (int(ceil(k / (1.0 * threads_per_block[0])))))

    pruneMultipleGPU [number_of_blocks, threads_per_block] (ci_d, k, min_support)

    ci_d.copy_to_host(ci_h)
    print "Outer Mask = ", ci_h.reshape(4, 4)

    ci_hn = np.zeros(k, dtype=np.int32)
    ci_dn = cuda.to_device(ci_hn)

    combinationsAvailable [threads_per_block, number_of_blocks] (ci_d, ci_dn, k)

    ci_dn.copy_to_host(ci_hn)

    print list(ci_hn)

    ci_hnx = np.empty(k, dtype=np.int32)
    ci_dnx = cuda.to_device(ci_hnx)

    preScan(ci_dnx, ci_dn, k)

    ci_dnx.copy_to_host(ci_hnx)
    print list(ci_hnx)

    sparseM_h = np.empty(ci_hnx[-1] * 3, dtype=np.uint32)
    sparseM_d = cuda.to_device(sparseM_h)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(k / (1.0 * threads_per_block[0]))), 1)

    convert2Sparse [threads_per_block, number_of_blocks] (ci_d, ci_dnx, sparseM_d, k)

    sparseM_d.copy_to_host(sparseM_h)

    sparseM_h = sparseM_h.reshape(3, ci_hnx[-1])
    print sparseM_h

    li_cpu = list(li_h)#[randint(10, 99) for i in range(0, 10000)]#list(li_h)

    t3 = time()
    selfJoinCPU(li_cpu)
    t4 = time()
    print t4 - t3

if __name__ == "__main__":
    test_apriori()