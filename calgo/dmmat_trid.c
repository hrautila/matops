
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_axpy.h"
#include "inner_ddot.h"

/*
  A: N*N, lower          X: N*1
     A00 | a01 | A02       x0
     ---------------      ----   
     a10 | a11 | a12       x1
     ---------------      ----
     A20 | a21 | A22       x2

   i = n: dimensions,
       a11 = 1*1, a10 = 1*n, a01 = n*1, a12 = N-(n+1)*1, A00 = n*n
       x1  = 1*1, x0  = n*1, x2  = N-(n+1)*1  

   A02, a12, a01 are zeros, A00, A22 are lower tridiagonal

 */

// Functions here implement various versions of TRMV operation.

// Calculates backward a diagonal block and updates Xc values from last to first.
// Updates are calculated in breadth first manner by with successive AXPY operations.
static void
_dmmat_trid_axpy_backward(double *Bc, const double *Ac, int unit,
                          int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *Bcl, *b0, *b1;
  register const double *Ar, *Acl;

  // diagonal matrix of nRE rows/cols and vector X of length nRE
  // move to point to last column and last entry of X.
  Acl = Ac + (nRE-1)*ldA;
  Bcl = Bc + nRE-1;

  // xr is the current X element, Ar is row in current A column.
  for (i = nRE; i > 0; i--) {
    Ar = Acl + i - 1; // move on diagonal
    b0 = Bcl;
    for (j = 0; j < nC; j++) {
      // update all b-values below with the current A column and current X
      _inner_daxpy(b0+1, Ar+1, b0, 1.0, nRE-i);
      b0[0] = unit ? b0[0] : b0[0]*Ar[0];
      b0 += ldB;
    }
    // previous row in B, previous column in A 
    Bcl--;
    Acl -= ldA;
  }
}

// Calculates backward a diagonal block and updates Xc values from last to first.
// Updates are calculated in depth first manner with DOT operations.
static void
_dmmat_trid_dot_backward(double *Bc, const double *Ac, int unit,
                         int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *b1, *Bcl;
  double xtmp;
  register const double *Ar, *Acl;

  // lower diagonal matrix (transposed) of nRE rows/cols and matrix B of size nRE*nC
  Acl = Ac + (nRE-1)*ldA;
  Bcl = Bc + nRE - 1;

  // xr is the current X element, Ar is row in current A column.
  for (i = 0; i < nRE; i++) {
    //Ar = Ac + i; // move on diagonal
    b1 = Bcl;
    b0 = Bc;
    for (j = 0; j < nC; j++) {
      // update all x-values below with the current A column and current X
      xtmp = unit ? b1[0] : 0.0;
      _inner_ddot(&xtmp, Acl, b0, 1.0, nRE-unit-i);
      b1[0] = xtmp;
      b0 += ldB;
      b1 += ldB;
    }

    // previous row in B, previous column in A 
    Bcl--;
    Acl -= ldA;
  }
}

// Calculate forward a diagonal block and updates Xc values from first to last.
static void
_dmmat_trid_axpy_forward(double *Bc, const double *Ac, double unit,
                         int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *Br;
  register const double *Ar;
  double *Broot = Bc;

  // xr is the current X element, Ar is row in current A column.
  Br = Bc;
  for (i = 0; i < nRE; i++) {
    b0 = Br;
    Bc = Broot;
    // update all previous x-values with current A column and current X
    for (j = 0; j < nC; j++) {
      _inner_daxpy(Bc, Ac, b0, 1.0, i);
      Ar = Ac + i;
      b0[0] = unit ? b0[0] : b0[0]*Ar[0];
      b0 += ldB;
      Bc += ldB;
    }
    //printf("B: i=%d\n", i); print_tile(Broot, ldB, nRE, nC);
    // next B row, next column in A 
    Br++;
    Ac += ldA;
  }
}

// Calculate forward a diagonal block and updates Xc values from first to last.
static void
_dmmat_trid_dot_forward(double *Bc, const double *Ac, int unit,
                        int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *b1, *Bcr;
  double xtmp;
  register const double *Ar;
  Bcr = Bc;
  for (i = 0; i < nRE; i++) {
    Ar = Ac + i + unit;
    b0 = Bcr;
    //b1 = Bcr;
    // update all previous B-values with current A column and current b
    for (j = 0; j < nC; j++) {
      //printf("i=%d, j=%d\n", i, j);
      xtmp = unit ? b0[0] : 0.0;
      _inner_ddot(&xtmp, Ar, b0, 1.0, nRE-unit-i);
      b0[0] = xtmp;
      b0 += ldB;
      //b1 += ldB;
    }
    // next row in B, next column in A 
    Bcr++;
    Ac += ldA;
  }
}

//extern void memset(void *, int, size_t);

// X = A*X; unblocked version
void dmmat_trid_unb(mdata_t *B, const mdata_t *A, int flags, int N, int S, int L)
{
  // indicates if diagonal entry is unit (=1.0) or non-unit.
  int unit = flags & MTX_UNIT ? 1 : 0;
  double *Bc = &B->md[S*B->step];
  
  if (flags & MTX_UPPER) {
    if (flags & MTX_TRANSA) {
      _dmmat_trid_dot_backward(Bc, A->md, unit, B->step, A->step, N, L-S);
    } else {
      _dmmat_trid_axpy_forward(Bc, A->md, unit, B->step, A->step, N, L-S);
    }
  } else {
    if (flags & MTX_TRANSA) {
      _dmmat_trid_dot_forward(Bc, A->md, unit, B->step, A->step, N, L-S);
    } else {
      _dmmat_trid_axpy_backward(Bc, A->md, unit, B->step, A->step, N, L-S);
    }
  }
}



// Local Variables:
// indent-tabs-mode: nil
// End:
