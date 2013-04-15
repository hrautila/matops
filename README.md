matops
======

Matrix operations. Almost complete implementation of BLAS level 1, 2 and 3 routines. All computation is in place. The implementation supports matrix views (submatrices of larger matrices) and parallel execution of matrix operations in multiple threads. Some functions already allow parallel execution for better performance.

Supported functionality is:

  Level 3

  Mult(C, A, B, alpha, beta, flags)       General matrix-matrix multiplication  (GEMM)
  MultSymm(C, A, B, alpha, beta, flags)   Symmetric matrix-matrix multipication (SYMM)
  MultTrm(B, A, alpha, flags) 	 	  Tridiagonal matrix-matrix multiplication (TRMM)  
  Solve(B, A, alpha, flags)		  Tridiagonal solve for multiple RHS (TRSM)
  RankUpdateSym(C, A, alpha, flags)	  Symmetric matrix rank update (SYRK)
  RankUpdate2Sym(C, A, B, alpha, flags)	  Symmetric matrix rank 2 update (SYR2K)

  Level 2

  MVMult(X, A, Y, alpha, beta, flags)	  General matrix-vector multiplcation (GEMV)
  MVRankUpdate(A, X, Y, alpha, flags)	  General matrix rank update (GER)
  MVRankUpdateSym(A, X, alpha, flags)	  Symmetric matrix rank update (SYR)
  MVRankUpdate2Sym(A, X, Y, alpha, flags) Symmetric matrix rank 2 update (SYR2)
  MVSolve(X, A, alpha, flags)		  Tridiagonal solve (TRSV)
  MVMultTrm(X, A, flags)		  Tridiagonal matrix-vector multiplication (TRSV)

  Level 1

  Norm2(X, Y)				  Vector norm
  Dot(X, Y)				  Inner product
  Swap(X, Y)				  Vector-vector swap
  InvScale(X, alpha)			  Inverse scaling of X
  Scale(X, alpha)			  Scaling of X 

  Lapack
  
  DecomposeLUnoPiv(A)			  LU decomposition with out pivoting
  DecompuseLU(A, pivots)		  LU decomposition with pivoting (GETRF)

This is still WORK IN PROGRESS. Consider this as beta level code, at best. 

Overall performance is comparable to ATLAS BLAS library. Some performance testing programs are in test subdirectory. Running package and performace tests requires github.com/hrautila/linalg packages as results are compared to existing BLAS/LAPACK implementation.

See the Wiki pages for some additional information. 
