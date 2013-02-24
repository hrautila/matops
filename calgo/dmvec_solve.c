
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_vec_axpy.h"

/*
  A: N*N, lower          X: N*1
     A00 | a01 | A02       x0
     ---------------      ----   
     a10 | a11 | a12       x1
     ---------------      ----
     a20 | a12 | A22       x2

   i = n: dimensions,
       a11 = 1*1, a10 = 1*n, a01 = n*1, a12 = N-(n+1)*1, A00 = n*n
       x1  = 1*1, x0  = n*1, x2  = N-(n+1)*1  

   A02, a12 are zeros, A00, A22 are lower tridiagonal

   if A diagonal is non-UNIT
      x1 = (x1 - a10*x0) / a11
   if a diagonal is UNIT
      x1 = x1 - a10*x0

   Operation a10*x0 is a DOT operation, and results 2n memory reads and 1 write.
   Vector a10 is row vector with elements N elements a part and x0 is column vector.
   Accessing x[i+1] is moving in memory direction and is likely a cache-hit. Accesing
   a10[i+1] is a memory reference to N*sizeof(double) bytes far and a likely cache-miss.
   
   A possible performace optimization: vector Y with size of X, initially zero.
   Y: 
      y0
     ----
      y1
     ----
      y2

   when i == n: y0 is n*1, y1 is 1*1 and y2 is N-(n+1)*1
   and y1 == a10*x0,

   x1 = (x1 - y1) / a11 or x1 = x1 - y1 if A is UNIT diagonal
   y2 = y2 + x1*a12
   
   a12 is column vector and a12[i+1] is likely to be cache-hit. A drawback is that
   we need write y memory locations N*N/2 times. Accessing vector x generates n memory
   reads and writes. 
 */

// solves backward a diagonal block and updates Xc values.  Yc vector
// contains per row accumulated values for already solved X's. 
static void
_dmvec_solve_backward(double *Xc, const double *Ac, double *Yc, int unit, 
                      int incX, int ldA, int incY, int nRE)
{
  // Y is 
  register int i;
  double *xr, *yr, xtmp;
  const double *Ar, *Acl;

  // upper diagonal matrix of nRE rows/cols and vector X, Y of length nRE
  // move to point to last column and last entry of X.
  Acl = Ac + (nRE-1)*ldA;
  Ar = Acl;
  xr = Xc + (nRE-1)*incX;
  yr = Yc + (nRE-1)*incY;

  // xr is the current X element, Ar is row in current A column.
  for (i = nRE-1; i >= 0; i--) {
    Ar = Acl + i;                // move on diagonal
    xtmp = xr[0] - yr[0];
    xr[0] = unit ? xtmp : xtmp/Ar[0];

    // update all y-values with in current column (i is the count above current row)
    _inner_vec_daxpy(Yc, incY, Acl, xr, incX, 1.0, i);
    // previous X, previous Y,  previous column in A 
    xr  -= incX;
    yr  -= incY;
    Acl -= ldA;
  }
}

// solves forward a diagonal block and updates Xc values.  Yc vector
// contains per row accumulated values for already solved X's. 
static void
_dmvec_solve_forward(double *Xc, const double *Ac, double *Yc, int unit,
                     int incX, int ldA, int incY, int nRE)
{
  // Y is 
  register int i;
  double *xr, *yr, xtmp;
  const double *Ar;

  // lower diagonal matrix of nRE rows/cols and vector X, Y of length nRE
  Ar = Ac;
  xr = Xc;
  yr = Yc;

  // xr is the current X element, Ar is row in current A column.
  for (i = 0; i < nRE; i++) {
    Ar = Ac + i;                // move on diagonal
    xtmp = xr[0] - yr[0];
    xr[0] = unit ? xtmp : xtmp/Ar[0];
    // update all y-values with in current column
    yr += incY;
    Ar ++;
    _inner_vec_daxpy(yr, incY, Ar, xr, incX, 1.0, nRE-i-1);
    // next X, next column in A 
    xr += incX;
    Ac += ldA;
  }
}

extern void memset(void *, int, size_t);

#define MAX_VEC_NB 256

// X = A(-1)*X
void dmvec_solve_unb(mvec_t *X, const mdata_t *A, int flags, int N)
{
  int i, nI;
  mvec_t Y;
  double cB[MAX_VEC_NB] __attribute__((aligned(64)));
  double *ybuf = (double *)0;
  int unit = flags & MTX_UNIT ? 1 : 0;
  Y.md = cB;
  Y.inc = 1;

  if (N > MAX_VEC_NB) {
    Y.md = (double *)calloc(N, sizeof(double));
    if (! Y.md) {
      return;
    }
  }

  memset(Y.md, 0, N*sizeof(double));
  if (flags & MTX_LOWER) {
    _dmvec_solve_forward(X->md, A->md, Y.md, unit, X->inc, A->step, 1, N);
  } else {
    _dmvec_solve_backward(X->md, A->md, Y.md, unit, X->inc, A->step, 1, N);
  }
  if (N > MAX_VEC_NB) {
    free(Y.md);
  }
}

/*
  A: N*N, lower          X: N*1
     A00 | A01 | A02       X0    Y0
     ---------------      ----  ----
     A10 | A11 | A12       X1    Y1
     ---------------      ----  ----
     A20 | A12 | A22       X2    Y2

   i = k: dimensions,
       A11 = n*n, A10 = n*k, A01 = k*n, A12 = N-(n+k)*n, A00 = k*k
       X1  = n*1, X0  = k*1, X2  = N-(n+k)*1  
       Y1  = n*1, Y0  = k*1, Y2  = N-(n+k)*1  

   A02, A12 are zeros, A00, A22 are lower tridiagonal


   Y0 = Y0 + A10*X0
   solve_forward_unb(X1, A11, Y0)

*/
void dmvec_solve_blocked(mvec_t *X, const mdata_t *A, int flags, int N, int NB)
{
  int i, nI, nR;
  mvec_t Y0;
  mdata_t A10;
  double cB[MAX_VEC_NB] __attribute__((aligned(16)));
  int unit = flags & MTX_UNIT ? 1 : 0;
  Y0.md = cB;
  Y0.inc = 1;

  if (NB <= 0) {
    NB = 68;
  }

  memset(Y0.md, 0, sizeof(cB));
  if (flags & MTX_LOWER) {
    nR = 0;
    for (i = 0; i < N; i += NB) {
      nI = N - i < NB ? N - i : NB;
      if (i > 0) {
        // calculate Y0 = Y0 + A10*X0
        A10.md = &A->md[nR];
        A10.step = A->step;
        //printf("A10:\n"); print_tile(A10.md, A10.step, nI, nR);
        dmult_gemv_blocked(&Y0, &A10, X, 1.0, 1.0, 0, 0, nR, 0, nI, 0, 0);
        //printf("Y0:\n"); print_tile(Y0.md, 1, nR, nI);
      }
      // solve forward using Y values 
      _dmvec_solve_forward(&X->md[i], &A->md[i*A->step+i], Y0.md, unit, X->inc, A->step, 1, nI);
      nR += nI;
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
