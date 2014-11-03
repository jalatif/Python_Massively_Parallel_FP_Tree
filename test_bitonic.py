__author__ = 'manshu'

from numba import *
from numbapro import cuda, cudadrv
import numpy as np
from random import randint
from time import time
from math import ceil

NUM_ELEMENTS = 16
BLOCK_SIZE = 4

def test_bitonic_sortCPU(b1, b2):
    ans = []
    if len(b1) == 1:
        if b1[0] > b2[0]:
            return [b2[0], b1[0]]
        else:
            return [b1[0], b2[0]]

    s1 = []
    s2 = []
    for i in range(0, len(b1)):
        s1.append(min(b1[i], b2[i]))
        s2.append(max(b1[i], b2[i]))
    print s1, s2
    split_point = len(s1) / 2
    ans += test_bitonic_sortCPU(s1[0:split_point], s1[split_point:])
    ans += test_bitonic_sortCPU(s2[0:split_point], s2[split_point:])

    return ans

def bitonicSort():

if __name__=="__main__":


    b1 = [1, 11, 21, 31]
    b2 = [2, 5, 10, 40]
    b2.reverse()


    for i in range(0, 4):


    print test_bitonic_sortCPU(b1, b2)

