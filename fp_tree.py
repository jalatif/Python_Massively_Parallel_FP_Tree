#!/home/manshu/Softwares/anaconda/bin/python

__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil

MAX_TRANSACTIONS = 1000000
MAX_SM_ITEMS = 12000
MAX_ITEMS_PER_TRANSACTIONS = 128
BLOCK_SIZE = 1024
MAX_UNIQUE_ITEMS = 124000

@jit(argtypes=[uint32[:], uint32[:], uint32[:], uint32, uint32], target='gpu')
def makeFlistGPU(d_offsets, d_transactions, d_flist, num_transactions, all_items_in_transactions):
    private_items = cuda.shared.array(MAX_SM_ITEMS, uint32)

    tx = cuda.threadIdx.x
    index = tx + cuda.blockDim.x * cuda.blockIdx.x
    location_x = 0
    for i in range(0, ceil(MAX_SM_ITEMS / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_SM_ITEMS:
            private_items[location_x] = 0

    cuda.syncthreads()

    item_ends = 0
    if index == (num_transactions - 1):
        item_ends = all_items_in_transactions
    elif index < (num_transactions - 1):
        item_ends = d_offsets[index + 1]
    else:
        item_ends = 0

    for i in range(d_offsets[index], item_ends):
        if d_transactions[i] >= 0 and d_transactions[i] < MAX_SM_ITEMS:
            cuda.atomic.add(private_items, d_transactions[i], 1)
        elif d_transactions[i] >= MAX_SM_ITEMS and d_transactions[i] < MAX_UNIQUE_ITEMS:
            cuda.atomic.add(d_flist, d_transactions[i], 1)

    cuda.syncthreads()

    for i in range(0, ceil(MAX_SM_ITEMS / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_SM_ITEMS:
            cuda.atomic.add(d_flist, location_x, private_items[location_x])

def myprint(string):
    print string

def readFile(file_name):
    fp = open(file_name, 'r')

    transactions = np.empty(MAX_ITEMS_PER_TRANSACTIONS * MAX_TRANSACTIONS, dtype=np.uint32)
    offsets = np.empty(MAX_TRANSACTIONS + 1, dtype=np.uint32)
    offsets[0] = 0

    trans_id = 0
    lines = 0

    for line in fp:
        if lines >= (MAX_TRANSACTIONS - 1): break
        line = line.strip()
        words = line.split(' ')
        for word in words:
            transactions[trans_id] = int(word)
            trans_id += 1
        offsets[lines + 1] = offsets[lines] + len(words)
        lines += 1

    return offsets, transactions, lines, trans_id

def makeFlist(transactions_h, nt):
    flist = {} #np.zeros(MAX_SM_ITEMS, dtype=np.uint32)
    print "Length = ", nt
    for i in range(0, nt):
        item = transactions_h[i]
        if item < MAX_UNIQUE_ITEMS:
            if item not in flist:
                flist[item] = 1
            else:
                flist[item] += 1

    return flist

def make_fp_tree():
    #### Allocate host memory
    offsets, transactions, num_transactions, all_items_in_transactions = readFile("data.txt")
    print num_transactions, all_items_in_transactions

    flist = np.zeros(MAX_UNIQUE_ITEMS, dtype=np.uint32)

    #### Allocate and initialize GPU/Device memory
    d_offsets = cuda.to_device(offsets)
    d_transactions = cuda.to_device(transactions)
    d_flist = cuda.to_device(flist)
    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(num_transactions / (1.0 * threads_per_block[0])) + 1, 1)

    t1 = time()
    makeFlistGPU [number_of_blocks, threads_per_block] (d_offsets, d_transactions, d_flist, num_transactions, all_items_in_transactions)
    cuda.synchronize()
    t2 = time()

    d_flist.copy_to_host(flist)
    cuda.synchronize()
    #
    # for i in range(0, MAX_UNIQUE_ITEMS):
    #     print i, flist[i]

    t3 = time()
    flist_cpu = makeFlist(transactions, all_items_in_transactions)
    t4 = time()
    #

    match = 1
    for i in range(1, MAX_UNIQUE_ITEMS):
        if i not in flist_cpu and flist[i] == 0:
            continue
        #print i, flist[i], flist_cpu[i]
        if flist[i] != flist_cpu[i]:
            match = -1
            break
    if match == 1:
        print "Test Passed"
    else:
        print "Test Failed"

    print "Number of transactions = ", num_transactions
    print "All items in transactions = ", all_items_in_transactions
    print "GPU time = ", t2 - t1
    print "CPU TIME = ", t4 - t3

make_fp_tree()