# -*- coding: utf-8 -*-
# @Time    : 2018/9/8 10:24
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : test.py
# @Software: PyCharm

from collections import Counter
c= Counter()
L1 = ['A','A','A','B','B',',E','E',',']
for i in L1:
    c[i] +=1
print(c)
print('--------------------')
c2 = Counter(L1)
print(c2)