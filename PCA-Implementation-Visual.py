import numpy as np
import pylab

# x = [[1, 2, 3], [1, 69, 68], [1, 68, 67], [1, 67, 66], [1, 3, 4], [1, 66, 65], [1, 4, 5]]
x = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]

x = np.asarray(x)
xtran = x.transpose()
xy = np.zeros(10)
xx = np.zeros(10)
for i in range(len(x)):
    xy[i-1] = x[i-1][0]
    xx[i-1] = x[i-1][1] 
pylab.plot(xtran[1],xtran[0], 'r+')
pylab.show()
# y = [1, 2, 2, 2, 1, 2, 1]

n = x.shape[1]  # number of parameters
m = x.shape[0]  # number of samples
r = 2  # number of parameters to reduce to

# x = np.eye(5)

# normalize the matrix
for i in range(0, n):
    col_sum = 0
    for j in range(0, m):
        col_sum += x[j][i]
    mean = col_sum / m
    for j in range(0, m):
        x[j][i] = x[j][i] - mean
# find the covariance matrix
cov_matrix = np.zeros((n, n))
for i in range(0, n):
    for j in range(i, n):
        col_1 = x[:, i]
        col_2 = x[:, j]
        entry = np.dot(col_1, col_2)
        cov_matrix[i][j] = entry
        cov_matrix[j][i] = entry
# find all eigenvalues and eigenvectors
e_vals, e_vecs = np.linalg.eig(cov_matrix)
# pick the top r to create the principal component matrix
for i in range(0, n - r):
    e_vecs = np.delete(e_vecs, -1, 1)
# multiply the pc matrix by the x matrix to get the final answer
final_data = np.dot(e_vecs.transpose(), x.transpose())
print(final_data)
#final_data = final_data.transpose()


pylab.plot(final_data[1],final_data[0], 'g.')
pylab.ylim(-2,2)
pylab.xlim(-2,2)
pylab.show()

pylab.plot(xtran[1],xtran[0], 'r+')
pylab.plot(final_data[1],final_data[0], 'g.')
pylab.ylim(-2,2)
pylab.xlim(-2,2)
pylab.show()