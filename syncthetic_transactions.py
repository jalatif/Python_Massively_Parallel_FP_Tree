__author__ = 'manshu'

import random

NUM_TRANSACTIONS = 100000
MAX_UNIQUE_ITEMS = 128
MAX_ITEMS_PER_TRANSACTION = 32

wF = open("syncthetic_data.txt", "w")

for row in range(0, NUM_TRANSACTIONS):
    line = ""
    for col in range(0, random.randint(1, MAX_ITEMS_PER_TRANSACTION)):
        line += str(random.randint(1, MAX_UNIQUE_ITEMS)) + " "
    line = line[:-1] + "\n"
    wF.write(line)
wF.close()
