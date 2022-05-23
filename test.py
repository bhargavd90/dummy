import pickle
import numpy as np
from collections import Counter

with open('resources/pos_dict', 'rb') as fp:
    pos_dict = pickle.load(fp)

for i in range(200):
    ctr_2 = Counter(pos_dict[i])
    most_common_keyword = ctr_2.most_common(5)
    print(most_common_keyword)
