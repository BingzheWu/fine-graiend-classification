import numpy as np
def GenerateBinomialTable(m):
    table = np.zeros((m+1, m+1), detype = np.float)
    for i in range(m+1):
        table[i, 0] = 1
    for i in range(1, m+1):
        for j in range(1, m+1):
            v = table[i-1, j] + table[i-1, j-1]
            table[i, j] = v
    return table