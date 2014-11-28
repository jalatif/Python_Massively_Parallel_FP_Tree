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

NUM_ELEMENTS = 32#1000000
MAX_UNIQUE_ITEMS = 16 # in range 1-6
MAX_PATTERN_SEARCH = 5
BLOCK_SIZE = 4
SM_SIZE = 2 * BLOCK_SIZE
MAX_ITEM_PER_SM = 16
MAX_TRANSACTIONS_PER_SM = 8
MAX_ITEMS_PER_TRANSACTIONS = 6
MAX_TRANSACTIONS = 4
SM_SHAPE = (MAX_TRANSACTIONS_PER_SM, MAX_ITEMS_PER_TRANSACTIONS)
SM_MAX_SIZE = 12288
MIN_SUPPORT = 2

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
    private_bin = cuda.shared.array(SM_MAX_SIZE, int32)
    tx = cuda.threadIdx.x
    index = tx + cuda.blockDim.x * cuda.blockIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    location_x = 0
    for i in range(0, ceil(SM_MAX_SIZE / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_UNIQUE_ITEMS and location_x < SM_MAX_SIZE:
            private_bin[location_x] = 0

    cuda.syncthreads()

    element = 0
    while index < num_elements:
        element = input_d[index]
        if element < SM_MAX_SIZE:
            cuda.atomic.add(private_bin, element, 1)
        else:
            cuda.atomic.add(bins_d, element, 1)

        index += stride

    cuda.syncthreads()

    for i in range(0, ceil(SM_MAX_SIZE / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < MAX_UNIQUE_ITEMS and location_x < SM_MAX_SIZE:
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

@jit(argtypes=[int32[:], int32[:], int32, int32], target='gpu')
def selfJoinGPU(input_d, output_d, num_elements, power):
    tx = cuda.threadIdx.x
    #index = tx + cuda.blockIdx.x * cuda.blockDim.x
    start = cuda.blockIdx.x * MAX_ITEM_PER_SM

    sm1 = cuda.shared.array(MAX_ITEM_PER_SM, int32)
    sm2 = cuda.shared.array(MAX_ITEM_PER_SM, int32)

    actual_items_per_sm = num_elements - start

    if actual_items_per_sm >= MAX_ITEM_PER_SM:
        actual_items_per_sm = MAX_ITEM_PER_SM

    for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
        location_x = tx + i * BLOCK_SIZE
        if location_x < actual_items_per_sm and (start + location_x) < num_elements:
            sm1[location_x] = input_d[start + location_x]
        else:
            sm1[location_x] = 0

    cuda.syncthreads()


    for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
        loop_tx = tx + i * BLOCK_SIZE
        if loop_tx < actual_items_per_sm:
            for j in range(loop_tx + 1, actual_items_per_sm):
                if (sm1[loop_tx] / (10 ** power)) == (sm1[j] / (10 ** power)):
                    output_d[(start + loop_tx) * num_elements + (start + j)] = 0
                # else:
                #      output_d[(start + loop_tx) * num_elements + (start + j)] = -1


    if (cuda.blockIdx.x + 1) < ceil(num_elements / (1.0 * MAX_ITEM_PER_SM)):
        current_smid = 0
        for smid in range(cuda.blockIdx.x + 1, ceil(num_elements / (1.0 * MAX_ITEM_PER_SM))):
            actual_items_per_secondary_sm = num_elements - current_smid * MAX_ITEM_PER_SM - start - MAX_ITEM_PER_SM
            if actual_items_per_secondary_sm > MAX_ITEM_PER_SM:
                actual_items_per_secondary_sm = MAX_ITEM_PER_SM

            for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
                location_x = tx + i * BLOCK_SIZE
                if location_x < actual_items_per_secondary_sm and (current_smid * MAX_ITEM_PER_SM + start + location_x) < num_elements:
                    sm2[location_x] = input_d[(current_smid + 1) * MAX_ITEM_PER_SM + start + location_x]
                else:
                    sm2[location_x] = 0
            cuda.syncthreads()

            for i in range(0, ceil(MAX_ITEM_PER_SM / (1.0 * BLOCK_SIZE))):
                loop_tx = tx + i * BLOCK_SIZE
                if loop_tx < actual_items_per_sm:
                    j = 0
                    while j < actual_items_per_secondary_sm:
                        if (sm1[loop_tx] / (10 ** power)) == (sm2[j] / (10 ** power)):
                            output_d[(start + loop_tx) * num_elements + (current_smid + 1) * MAX_ITEM_PER_SM + start + j] = 0
                        # else:
                        #     output_d[(start + loop_tx) * num_elements + smid * MAX_ITEM_PER_SM + start + j] = -1
                        j += 1
            current_smid += 1

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

@jit(argtypes=[int32[:], int32[:], int32[:], int32, int32], target='gpu')
def convert2Sparse(input_d, offset_d, output_d, num_patterns, k):
    tx = cuda.threadIdx.x
    index = tx + cuda.blockIdx.x * cuda.blockDim.x

    if index < (k - 1):
        col_start = offset_d[index]
        #col_end = offset_d[index + 1]
        for i in range(0, k): ### Bug
            support_value = input_d[index * k + i]
            if support_value > 0:
                output_d[col_start] = index
                output_d[col_start + num_patterns] = i
                output_d[col_start + 2 * num_patterns] = support_value
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
        for word_id in range(0, len(words)):
            if word_id >= MAX_ITEMS_PER_TRANSACTIONS:
                print "Warning: Items in transactions exceeding MAX_ITEMS_PER_TRANSACTIONS"
                break
            word = words[word_id]
            try:
                word = int(word)
            except:
                print "Error: Wrong type of item in transaction. Exiting..."
                sys.exit(10)

            if int(word) >= MAX_UNIQUE_ITEMS:
                print "Warning: Item in transaction exceeds or equals MAX_UNIQUE_ITEMS"
                continue
            transactions[trans_id] = word
            trans_id += 1
        offsets[lines + 1] = offsets[lines] + min([len(words), MAX_ITEMS_PER_TRANSACTIONS])
        lines += 1

    return offsets, transactions, lines, trans_id

@jit(argtypes=[int32[:], int32[:], int32, int32, int32[:], int32[:], int32, int32[:], int32[:]], target='gpu')
def findHigherPatternFrequencyGPU(d_transactions, d_offsets, num_transactions, num_elements, dkeyIndex, dMask, num_patterns, api_d, iil_d):
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
            vpat1 = dkeyIndex[loop_tx]
            vpat2 = dkeyIndex[last_seen]

            v_common_pat = vpat1 / 10
            vitem1 = vpat1 % 10
            vitem2 = vpat2 % 10

            item1 = api_d[iil_d[(vitem1-1) * 3 + 1]]
            item2 = api_d[iil_d[(vitem2-1) * 3 + 1]]

            pattern_items = cuda.shared.array(MAX_PATTERN_SEARCH, int32)
            flag_pattern_items = cuda.shared.array(MAX_PATTERN_SEARCH, int32)

            common_pat_start = iil_d[(v_common_pat-1) * 3 + 1]
            common_pat_length = iil_d[(v_common_pat-1) * 3 + 2]
            common_pat_end = common_pat_start + common_pat_length

            # if loop_tx == 0 and tx == 0:
            #     print -1
            #     print -1
            #     print v_common_pat
            #     print common_pat_start
            #     print common_pat_end
            #     print -1
            #     print -1
            for item_id in range(common_pat_start, common_pat_end):
                pattern_items[item_id - common_pat_start] = api_d[item_id]


            for tid in range(0, MAX_TRANSACTIONS_PER_SM):
                flag_item1 = False
                flag_item2 = False

                for titem in range(0, MAX_ITEMS_PER_TRANSACTIONS):
                    if Ts[tid, titem] == item1: flag_item1 = True
                    elif Ts[tid, titem] == item2: flag_item2 = True
                    for item_id in range(0, common_pat_length):
                        if Ts[tid, titem] == pattern_items[item_id]: flag_pattern_items[item_id] = 1

                pattern_flag = 1
                for item_id in range(0, common_pat_length):
                    pattern_flag = pattern_flag * flag_pattern_items[item_id]

                present_flag = flag_item1 and flag_item2

                if present_flag and (pattern_flag == 1):
                    cuda.atomic.add(dMask, loop_tx * num_patterns + last_seen, 1)

def test_apriori():

    offsets, transactions, num_transactions, num_elements = readFile("dummy.txt")
    print "Offset = ", offsets[:num_transactions]
    print "transactions = ", transactions[:num_elements]
    print "Num transactions = ", num_transactions
    print "Num elements = ", num_elements
    min_support = MIN_SUPPORT

    # to find number of max digits required to represent that many number of unique items

    power = 1
    while MAX_UNIQUE_ITEMS / (10 ** power) != 0:
        power += 1


    print "Power = ", power

    t = [item for item in transactions.tolist()]

    if num_elements > NUM_ELEMENTS:
        print "Error: Elements exceeding NUM_ELEMENTS. Exiting..."
        sys.exit(12)

    input_h = np.array(t, dtype=np.int32)
    print "Input transactions = ", list(input_h)
    ci_h = np.zeros(MAX_UNIQUE_ITEMS, dtype=np.int32)
    li_h = np.empty(MAX_UNIQUE_ITEMS, dtype=np.int32)

    input_d = cuda.to_device(input_h)
    ci_d = cuda.to_device(ci_h)
    li_d = cuda.device_array(MAX_UNIQUE_ITEMS, dtype=np.int32)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(NUM_ELEMENTS / (1.0 * threads_per_block[0]))), 1)#((NUM_ELEMENTS / threads_per_block[0]) + 1, 1)

    histogramGPU [number_of_blocks, threads_per_block] (input_d, ci_d, num_elements)
    #cuda.synchronize()

    ci_d.copy_to_host(ci_h)
    print "Ci_H Histogram result = ", ci_h # support count for each item

    number_of_blocks = (int(ceil(MAX_UNIQUE_ITEMS / (1.0 * threads_per_block[0]))), 1)
    pruneGPU [number_of_blocks, threads_per_block] (ci_d, MAX_UNIQUE_ITEMS, min_support)
    cuda.synchronize()

    ci_d.copy_to_host(ci_h)
    print "Keys = ", [i for i in range(0, len(ci_h))]
    print "Ci_H Pruning result = ", ci_h # support count for each item

    # calculate concise list of items satisfying min support
    k = 0 # number of items whose sup_count > min_support
    for j in range(0, len(ci_h)):
        if ci_h[j] != 0:
            li_h[k] = j
            k += 1

    print "LI_H = ", list(li_h)[:k]  #items whose support_count > min_support

    print "K(num_items_with_good_sup_count = ", k

    #k = 102
    ci_h = np.array([-1 for i in range(0, k ** 2)], dtype=np.int32)
    ci_d = cuda.to_device(ci_h)

    li_h = np.array(sorted([randint(10, 99) for i in range(0, k)]), dtype=np.int32)
    #tli_h = np.array([i for i in range(1, k + 1)], dtype=np.int32)

    t1 = time()
    li_d = cuda.to_device(li_h)
    number_of_blocks = (int(ceil(k / (1.0 * MAX_ITEM_PER_SM))), 1)
    selfJoinGPU [number_of_blocks, threads_per_block](li_d, ci_d, k, power)

    li_d.copy_to_host(li_h)
    ci_d.copy_to_host(ci_h)
    t2 = time()

    # f = open('join.txt', 'w')
    #
    # for i in range(0, k):
    #     line = ""
    #     for j in range(0, k):
    #         line += str(ci_h[k * i + j]) + " "
    #     f.write(line + "\n")
    #
    # f.close()
    #ci_h = ci_h.reshape(k, k)

    print "Initial Mask = ", ci_h.reshape(k, k)

    print "Self joining time = ", (t2 - t1)

    sys.exit(0)

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
    print "Final Mask = ", ci_h.reshape(k, k)
    d_transactions.copy_to_host(transactions)

    print transactions[:num_elements]

    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    number_of_blocks = ((int(ceil(k / (1.0 * threads_per_block[0])))), (int(ceil(k / (1.0 * threads_per_block[0])))))

    pruneMultipleGPU [number_of_blocks, threads_per_block] (ci_d, k, min_support)

    ci_d.copy_to_host(ci_h)
    print "Outer Mask = ", ci_h.reshape(k, k)

    ci_hn = np.zeros(k, dtype=np.int32)
    ci_dn = cuda.to_device(ci_hn)

    combinationsAvailable [threads_per_block, number_of_blocks] (ci_d, ci_dn, k)

    ci_dn.copy_to_host(ci_hn)

    print "Ci_hn = ", list(ci_hn)

    ci_hnx = np.empty(k, dtype=np.int32)
    ci_dnx = cuda.to_device(ci_hnx)

    preScan(ci_dnx, ci_dn, k)

    ci_dnx.copy_to_host(ci_hnx)
    num_patterns = ci_hnx[-1]
    print "Ci_hnx = ", list(ci_hnx)

    sparseM_h = np.empty(ci_hnx[-1] * 3, dtype=np.uint32)
    sparseM_d = cuda.to_device(sparseM_h)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(k / (1.0 * threads_per_block[0]))), 1)

    convert2Sparse [threads_per_block, number_of_blocks] (ci_d, ci_dnx, sparseM_d, num_patterns, k)

    sparseM_d.copy_to_host(sparseM_h)

    # sparseM_h = sparseM_h.reshape(3, num_patterns)
    print sparseM_h.reshape(3, num_patterns)

    patterns = {}
    for i in range(0, num_patterns):
        item1 = sparseM_h[i]
        item2 = sparseM_h[i + num_patterns]
        support = sparseM_h[i + 2 * num_patterns]
        patterns[tuple(sorted([li_h[item1], li_h[item2]]))] = support
    print patterns

    new_modulo_map = {}
    index_id = 1

    actual_pattern_items = []
    index_items_lookup = []

    #patterns = {(2, 3, 5) : 1, (2, 3, 6) : 1, (2, 3, 7) : 1, (2, 4, 5) : 1, (2, 4, 7) : 1, (3, 5, 7) : 1}
    for pattern in patterns:
        if pattern[:-1] not in new_modulo_map:
            new_modulo_map[pattern[:-1]] = index_id
            prev_len = len(actual_pattern_items)
            pattern_len = len(pattern[:-1])
            actual_pattern_items += pattern[:-1]
            index_items_lookup += [index_id, prev_len, pattern_len]
            index_id += 1

        if (pattern[-1],) not in new_modulo_map:
            new_modulo_map[(pattern[-1],)] = index_id
            prev_len = len(actual_pattern_items)
            pattern_len = len([pattern[-1]])
            actual_pattern_items += [pattern[-1]]
            index_items_lookup += [index_id, prev_len, pattern_len]
            index_id += 1


    print "Actual pattern items = ", actual_pattern_items
    print "Index lookup = ", index_items_lookup
    print new_modulo_map

    new_patterns = []
    for pattern in patterns:
        new_patterns.append((new_modulo_map[pattern[:-1]], new_modulo_map[(pattern[-1],)]))
    print new_patterns

    new_new_pattern = []
    for pattern in new_patterns:
        new_new_pattern.append(pattern[0] * 10 ** power + pattern[1])

    new_new_pattern.sort()
    print new_new_pattern

    k = len(new_new_pattern)

    li_h = np.array(new_new_pattern, dtype=np.int32)

    ci_h = np.array([-1 for i in range(0, k ** 2)], dtype=np.int32)
    ci_d = cuda.to_device(ci_h)


    #li_h = np.array(sorted([randint(10, 99) for i in range(0, k)]), dtype=np.int32)

    t1 = time()
    li_d = cuda.to_device(li_h)
    number_of_blocks = (int(ceil(k / MAX_ITEM_PER_SM)), 1)
    selfJoinGPU [number_of_blocks, threads_per_block](li_d, ci_d, k, power)

    li_d.copy_to_host(li_h)
    ci_d.copy_to_host(ci_h)

    api_h = np.array(actual_pattern_items, dtype=np.int32)
    iil_h = np.array(index_items_lookup, dtype=np.int32)

    api_d = cuda.to_device(api_h)
    iil_d = cuda.to_device(iil_h)


    t2 = time()
    print "LI_H = ", li_h
    print "Initial Mask = ", ci_h

    threads_per_block = (BLOCK_SIZE, 1)
    #number_of_blocks = (1, 1) #(int(num_transactions / (1.0 * threads_per_block[0])) + 1, 1)
    number_of_blocks = (int(ceil(num_transactions / (1.0 * MAX_TRANSACTIONS_PER_SM))), 1)

    print "Num transactions = ", num_transactions
    print "Num patterns = ", k
    print "index = ", li_h
    findHigherPatternFrequencyGPU [number_of_blocks, threads_per_block] (d_transactions, d_offsets, num_transactions, num_elements, li_d, ci_d, k, api_d, iil_d)
    cuda.synchronize()
    ci_d.copy_to_host(ci_h)
    print "Final Mask = ", ci_h
    d_transactions.copy_to_host(transactions)

    print transactions[:num_elements]

    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    number_of_blocks = ((int(ceil(k / (1.0 * threads_per_block[0])))), (int(ceil(k / (1.0 * threads_per_block[0])))))

    pruneMultipleGPU [number_of_blocks, threads_per_block] (ci_d, k, min_support)

    ci_d.copy_to_host(ci_h)
    print "Outer Mask = ", ci_h.reshape(k, k)
    print "K = ", k

    ci_hn = np.zeros(k, dtype=np.int32)
    ci_dn = cuda.to_device(ci_hn)

    combinationsAvailable [threads_per_block, number_of_blocks] (ci_d, ci_dn, k)

    ci_dn.copy_to_host(ci_hn)

    print "Ci_hn = ", list(ci_hn)

    ci_hnx = np.empty(k, dtype=np.int32)
    ci_dnx = cuda.to_device(ci_hnx)

    preScan(ci_dnx, ci_dn, k)

    ci_dnx.copy_to_host(ci_hnx)
    num_patterns = ci_hnx[-1]
    print list(ci_hnx)

    sparseM_h = np.empty(ci_hnx[-1] * 3, dtype=np.uint32)
    sparseM_d = cuda.to_device(sparseM_h)

    threads_per_block = (BLOCK_SIZE, 1)
    number_of_blocks = (int(ceil(k / (1.0 * threads_per_block[0]))), 1)
    print "K = ", k

    convert2Sparse [threads_per_block, number_of_blocks] (ci_d, ci_dnx, sparseM_d, num_patterns, k)

    sparseM_d.copy_to_host(sparseM_h)

    # sparseM_h = sparseM_h.reshape(3, num_patterns)
    print sparseM_h.reshape(3, num_patterns)

    patterns = {}
    for i in range(0, num_patterns):
        item1 = sparseM_h[i]
        item2 = sparseM_h[i + num_patterns]
        support = sparseM_h[i + 2 * num_patterns]
        patterns[tuple(sorted([li_h[item1], li_h[item2]]))] = support
    print patterns

    actual_patterns = {}

    for pattern in patterns:
        v_common_pat = pattern[0] / 10
        vitem1 = pattern[0] % 10
        vitem2 = pattern[1] % 10

        item1 = actual_pattern_items[index_items_lookup[(vitem1-1) * 3 + 1]]
        item2 = actual_pattern_items[index_items_lookup[(vitem2-1) * 3 + 1]]


        common_pat_start = index_items_lookup[(v_common_pat-1) * 3 + 1]
        common_pat_length = index_items_lookup[(v_common_pat-1) * 3 + 2]
        common_pat_end = common_pat_start + common_pat_length

        common_pattern = actual_pattern_items[common_pat_start:common_pat_end]

        pattern_key = tuple(common_pattern) + tuple(sorted([item1, item2]))
        actual_patterns[pattern_key] = patterns[pattern]

    print "L2 = ", actual_patterns


if __name__ == "__main__":
    test_apriori()