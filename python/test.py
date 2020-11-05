import timeit

setup = '''
import numpy as np
# from numba import jit

# @jit
def bubblesort(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp


original = np.arange(0.0, 10.0, 0.01, dtype='f4')
shuffled = original.copy()
np.random.shuffle(shuffled)

sorted = shuffled.copy()
# bubblesort(sorted)
# print(np.array_equal(sorted, original))
# sorted[:] = shuffled[:]; 
'''

print(timeit.timeit(setup=setup, stmt="bubblesort(sorted)", number=1))
