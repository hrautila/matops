matops
======

Matrix operations. Almost complete implementation of BLAS level 1, 2 and 3 routines for double precision floating point. All computation is in place. The implementation supports matrix views (submatrices of larger matrices) and parallel execution of matrix operations in multiple threads. Some functions already allow parallel execution for better performance.

Supported functionality is:

  Blas level 3

    Mult(C, A, B, alpha, beta, flags)           General matrix-matrix multiplication  (GEMM)
    MultSymm(C, A, B, alpha, beta, flags)       Symmetric matrix-matrix multipication (SYMM)
    MultTrm(B, A, alpha, flags)                 Tridiagonal matrix-matrix multiplication (TRMM)  
    Solve(B, A, alpha, flags)                   Tridiagonal solve with multiple RHS (TRSM)
    RankUpdateSym(C, A, alpha, beta,flags)      Symmetric matrix rank update (SYRK)
    RankUpdate2Sym(C, A, B, alpha, beta, flags) Symmetric matrix rank 2 update (SYR2K)

  Blas level 2

    MVMult(X, A, Y, alpha, beta, flags)         General matrix-vector multiplcation (GEMV)
    MVRankUpdate(A, X, Y, alpha, flags)         General matrix rank update (GER)
    MVRankUpdateSym(A, X, alpha, flags)         Symmetric matrix rank update (SYR)
    MVRankUpdate2Sym(A, X, Y, alpha, flags)     Symmetric matrix rank 2 update (SYR2)
    MVSolve(X, A, alpha, flags)                 Tridiagonal solve (TRSV)
    MVMultTrm(X, A, flags)                      Tridiagonal matrix-vector multiplication (TRMV)

  Blas level 1

    Norm2(X, Y)         Vector norm (NRM2)
    Dot(X, Y)           Inner product (DOT)
    Swap(X, Y)          Vector-vector swap (SWAP)
    InvScale(X, alpha)  Inverse scaling of X 
    Scale(X, alpha)     Scaling of X (SCAL)

  Lapack
  
    DecomposeLUnoPiv(A)       LU decomposition without pivoting
    DecomposeLU(A, pivots)    LU decomposition with pivoting (GETRF)
    DecomposeCHOL(A)          Cholesky decomposition (POTRF)

  Support functions

    TriL(A)                   Make A tridiagonal, lower 
    TriLU(A)                  Make A tridiagonal, lower, unit-diagonal 
    TriU(A)                   Make A tridiagonal, upper 
    TriUU(A)                  Make A tridiagonal, upper, unit-diagonal 

  Parameter functions

    BlockingParams(m,n,k)     Blocking size parameters for low-level functions
    NumWorkers(nwrk)          Number of threads for use in operations
    DecomposeBlockSize(nb)    Block size for blocked decomposition algorithms

This is still WORK IN PROGRESS. Consider this as beta level code, at best. 

Overall performance is compareable to ATLAS BLAS library. Some performance testing programs are in test subdirectory. Running package and performace tests requires github.com/hrautila/linalg packages as results are compared to existing BLAS/LAPACK implementation.

See the Wiki pages for some additional information. 
