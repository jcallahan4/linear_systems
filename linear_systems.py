# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""
import numpy as np
import scipy as sp
import time
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    #check for square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not invertible!")
    #get size of matrix
    n = A.shape[0]
    #create A
    A = A.astype(float)
    #reduce rows
    for k in range(n - 1):
        #subtract rows
        for j in range(k+1,n):
            c = A[j,k] / A[k,k]
            A[j,k+1:] = A[j,k+1:] - c * A[k,k+1:]
            A[j,k] = 0
        #set leading element to 1
        if k == n - 1:
            A[k,k] == 1
    return A

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    #create needed matrices
    A = A.astype(float)
    m,n = A.shape
    U = A.copy()
    L = np.identity(m)
    #row reduce matrix while recording the reduction in L
    for j in range(n):
        for i in range(j + 1, m):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] = U[i,j:] - L[i,j]*U[j,j:]
    return L,U


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    #create needed data structures
    A = A.astype(float)
    b = b.astype(float)
    m,n = A.shape[0], A.shape[1]
    L,U = lu(A)
    y = np.zeros(n)
    x = np.zeros(n)
    sum = 0
    #populate y with correct values
    for k in range(n):
        sum = 0
        for j in range(k):
            sum += L[k,j]*y[j]
        y[k] = b[k] - sum

    #populate x with correct values
    for k in range(m)[::-1]:
        sum = 0
        for j in range(k+1,m):
            sum += U[k,j]*x[j]
        x[k] = (y[k] - sum) / U[k,k]

    return x

# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    one_times = []
    two_times = []
    three_times =[]
    four_times = []

    domain = 2**np.arange(1, 13)

    # 1)
    for n in domain:
        #create matrices
        A = np.random.random((n,n))
        b = np.random.random(n)

        #time inversion
        start = time.time()
        Ainv = la.inv(A)
        X = Ainv @ b
        final = time.time() - start
        one_times.append(final)

        #time solve
        start = time.time()
        x = la.solve(A,b)
        final = time.time() - start
        two_times.append(final)

        #time lu factor and solve
        start = time.time()
        L,P = la.lu_factor(A)
        x = la.lu_solve((L,P), b)
        final = time.time() - start
        three_times.append(final)

        #time lu solve
        L,P = la.lu_factor(A)
        start = time.time()
        x = la.lu_solve((L,P),b)
        final = time.time() - start
        four_times.append(final)

    #plot times
    ax1 = plt.subplot(111)
    ax1.set_title("Linalg times")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")
    ax1.loglog(domain, one_times, 'g.-', linewidth = 2, markersize = 15, label = "la.inv", basey= 2, basex = 2)
    ax1.loglog(domain, two_times, 'k.-', linewidth = 2, markersize = 15, label = "la.solve", basey = 2, basex = 2)
    ax1.loglog(domain, three_times, 'r.-', linewidth =2, markersize = 15, label = "lu_solve and factor", basey = 2, basex = 2)
    ax1.loglog(domain, four_times, 'b.-', linewidth = 2, markersize = 15, label = "lu_solve", basey = 2, basex = 2)
    ax1.legend(loc = 'upper left')
    plt.show()

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    #Create B
    B = sparse.diags([1,-4,1], [-1,0,1], shape = (n,n)).toarray()
    #Create block diagonal matrix with B
    A_1 = sparse.block_diag([B] * n)
    #Create matrix with offset identities
    A_2 = sparse.diags([1,1], [-n,n], shape = (n**2, n**2))
    #Add them together to create A as desired
    A = A_1 + A_2

    return A


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    domain = 2**np.arange(1,7)
    start = 0
    final = 0
    CSR_times = []
    numpy_times = []
    for n in domain:
        #create A
        A = prob5(n)
        b = np.random.random(n**2)

        #cast to CSR
        ACSR = A.tocsr()
        #time CSR solve
        start = time.time()
        spla.spsolve(ACSR, b)
        final = time.time() - start
        CSR_times.append(final)

        #Cast to numpy
        Anumpy = A.toarray()
        start = time.time()
        #begin time
        la.solve(Anumpy, b)
        final = time.time() - start
        numpy_times.append(final)

    ax1 = plt.subplot(121)
    ax1.set_title("Times on Regular Scale")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")

    #Plot lists against domain
    ax1.loglog(domain**2, CSR_times,'g.-', linewidth = 2, markersize = 7, label = "CSR Times")
    ax1.loglog(domain**2, numpy_times,'b.-', linewidth = 2, markersize = 7, label = "Numpy Times")
    ax1.legend(loc = "upper right")

    plt.show()
