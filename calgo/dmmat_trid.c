
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
#include "inner_ddot_trans.h"

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

/*
  B = A*B; A is lower

    a00| 0 | 0   b00|b01
    a10|a11| 0   b10|b11
    a20|a21|a22  b20|b21

  b00 = a00*b00
  b01 = a00*b01
  b10 = a10*b00 + a11*b10
  b11 = a10*b01 + a11*b11
  b20 = a20*b00 + a21*b10 + a22*b20
  b21 = a20*b01 + a21*b11 + a22*b21

  --> work it backwards as b12 & b02 are not needed for b11, b01, ...

  Calculates backward a diagonal block and updates Xc values from last to first.
  Updates are calculated in breadth first manner by with successive AXPY operations.
 */
static void
_dmmat_trid_axpy_backward(double *Bc, const double *Ac, double alpha, int unit,
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

  for (i = nRE; i > 0; i--) {
    Ar = Acl + i - 1; // move on diagonal
    b0 = Bcl;
    for (j = 0; j < nC; j++) {
      // update all b-values below with the current A column and current X
      _inner_daxpy(b0+1, Ar+1, b0, alpha, nRE-i);
      b0[0] = alpha * (unit ? b0[0] : b0[0]*Ar[0]);
      b0 += ldB;
    }
    // previous row in B, previous column in A 
    Bcl--;
    Acl -= ldA;
  }
}

/*
  B = A*B; A is upper, trans A

    a00|a01|a02  b00|b01
     0 |a11|a12  b10|b11
     0 | 0 |a22  b20|b21

  b00 = a00*b00 + a01*b10 + a02*b20
  b01 = a00*b01 + a01*b11 + a02*b21
  b10 =           a11*b10 + a12*b20
  b11 =           a11*b11 + a12*b21
  b20 =                     a22*b20
  b21 =                     a22*b21

  --> work it backwards with DOT products

  Calculates backward a diagonal block and updates Xc values from last to first.
  Updates are calculated in depth first manner with DOT operations.
 */
static void
_dmmat_trid_dot_backward(double *Bc, const double *Ac, double alpha, int unit,
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

  for (i = 0; i < nRE; i++) {
    b1 = Bcl;
    b0 = Bc;
    for (j = 0; j < nC; j++) {
      // update all B-values below with the current A column and current X
      xtmp = unit ? alpha*b1[0] : 0.0;
      _inner_ddot(&xtmp, Acl, b0, alpha, nRE-unit-i);
      b1[0] = xtmp;
      b0 += ldB;
      b1 += ldB;
    }

    // previous row in B, previous column in A 
    Bcl--;
    Acl -= ldA;
  }
}

/*
    B = A*B; A is upper

      a00|a01|a02  b00|b01
       0 |a11|a12  b10|b11
       0 | 0 |a22  b20|b21

    b00 = a00*b00 + a01*b10 + a02*b20
    b10 =           a11*b10 + a12*b20
    b20 =                     a22*b20
    b01 = a00*b01 + a01*b11 + a02*b21
    b11 =           a11*b11 + a12*b21
    b21 =                     a22*b21

    --> work it forwards as b00, b01 not need for later elements; AXPY

 Calculate forward a diagonal block and updates Xc values from first to last.
*/

static void
_dmmat_trid_axpy_forward(double *Bc, const double *Ac, double alpha, int unit,
                         int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *Br;
  register const double *Ar;
  double *Broot = Bc;

  Br = Bc;
  for (i = 0; i < nRE; i++) {
    b0 = Br;
    Bc = Broot;
    // update all previous B-values with current A column and current B
    for (j = 0; j < nC; j++) {
      _inner_daxpy(Bc, Ac, b0, alpha, i);
      Ar = Ac + i;
      b0[0] = unit ? b0[0] : alpha*b0[0]*Ar[0];
      b0 += ldB;
      Bc += ldB;
    }
    // next B row, next column in A 
    Br++;
    Ac += ldA;
  }
}

// Calculate forward a diagonal block and current B-values from first to last.
static void
_dmmat_trid_dot_forward(double *Bc, const double *Ac, double alpha, int unit,
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
    // update all current B-value with current A column and following b
    for (j = 0; j < nC; j++) {
      xtmp = unit ? alpha*b0[0] : 0.0;
      _inner_ddot(&xtmp, Ar, b0, alpha, nRE-unit-i);
      b0[0] = xtmp;
      b0 += ldB;
    }
    // next row in B, next column in A 
    Bcr++;
    Ac += ldA;
  }
}

/*
  for B = B*A; A is [nC, nC], B is [nRE, nC]
  
    b00|b01|b02  a00|a01|a02
    b10|b11|b12   0 |a11|a12
                  0 | 0 |a22

    b00 = b00*a00
    b10 = b10*a00
    b01 = b00*a01 + a11*b01
    b11 = b10*a01 + a11*b11
    b02 = b00*a02 + a12*b01 + a22*b02
    b12 = b10*a02 + a12*b11 + a22*b12
    
    --> work it backwards as b12 & b02 are not needed for b11, b01, ...
*/
static void
_dmmat_trid_dot_bleft_backwd(double *Bc, const double *Ac, double alpha, int unit,
                             int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *Br, *Bcl;
  register const double *Ar, *Acl;
  double btmp;

  // last columns of A
  Acl = Ac + (nC-1)*ldA;
  Bcl = Bc + (nC-1)*ldB;
  for (j = nC; j > 0; j--) {
    Br = Bc;
    b0 = Bcl;
    // update all B-values with current A column and current B, AXPY
    for (i = 0; i < nRE; i++) {
      Ar = Acl;
      btmp = unit ? alpha*b0[0] : 0.0;
      // calculate dot-product following Ar column and Br row
      _inner_ddot_trans(&btmp, Ar, Br, alpha, j-unit, ldB);
      b0[0] = btmp;
      b0++;
      Br++;
    }
    // previous B column, previous A column
    Bcl -= ldB;
    Acl -= ldA;
  }
}

static void
_dmmat_trid_dot_bleft_fwd(double *Bc, const double *Ac, double alpha, int unit,
                          int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *Br, *Bcl;
  register const double *Ar, *Acl;
  double btmp;

  // columns of A
  Acl = Ac;
  Bcl = Bc;
  for (j = nC; j > 0; j--) {
    Br = Bcl;
    b0 = Bcl;
    // update all B-values with current A column and current B, AXPY
    for (i = 0; i < nRE; i++) {
      Ar = Acl + nC - j;
      btmp = 0.0;
      // calculate dot-product following Ar column and Br row
      _inner_ddot_trans(&btmp, Ar+unit, Br+unit, alpha, j-unit, ldB);
      b0[0] = unit ? btmp + alpha*b0[0] : btmp;
      b0++;
      Br++;
    }
    // next B column, next A column
    Bcl += ldB;
    Acl += ldA;
  }
}

/*
  for B = B*A.T; A.T is [nC, nC], B is [nRE, nC], 
  
    b00|b01|b02  a00|a01|a02
    b10|b11|b12   0 |a11|a12
                  0 | 0 |a22

    b00 = b00*a00 + a01*b01 + a02*b02
    b01 =           a11*b01 + a12*b02
    b02 =                     a22*b02
    b10 = b10*a00 + a01*b11 + a02*b12
    b11 =           a11*b11 + a12*b02
    b12 =                     a22*b12
    
    --> work it forward as b00 & b010 are not needed for b01, b11, ... with AXPY
*/
static void
_dmmat_trid_axpy_bleft_fwd(double *Bc, const double *Ac, double alpha, int unit,
                           int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *Br, *Bcl;
  register const double *Ar, *Acl;

  // columns of A
  Acl = Ac;
  Bcl = Bc;
  for (i = 0; i < nRE; i++) {
    Br = Bcl;
    b0 = Bcl;
    Acl = Ac;
    // update all B-values with current A column and current B, AXPY
    for (j = 0; j < nC; j++) {
      Ar = Acl + j;     // diagonal entry on A.
      // update preceding elements on B row
      _inner_axpy_trans(Bcl, Acl, b0, alpha, j, ldB);
      // update current element on B rows
      b0[0] = unit ? alpha*b0[0] : alpha*Ar[0]*b0[0];
      b0 += ldB;        // next element on B rows
      Acl += ldA;       // next column in A
    }
    // next B column, next A column
    Bcl++;
  }
}

/*
  for B = B*A.T; A.T is [nC, nC], B is [nRE, nC], 
  
    b00|b01|b02  a00| 0 | 0
    b10|b11|b12  a10|a11| 0
                 a20|a21|a22

    b00 = b00*a00
    b01 = b00*a10 + a11*b01
    b02 = b00*a20 + a21*b01 + a22*b02
    
    --> work it backward as b02 & b12 are not needed for b01, b11, ... with AXPY
*/
static void
_dmmat_trid_axpy_bleft_backwd(double *Bc, const double *Ac, double alpha, int unit,
                              int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  register double *b0, *Br, *Bcl;
  register const double *Ar, *Acl;

  // columns of A
  Bcl = Bc + (nC-1)*ldB;

  for (i = 0; i < nRE; i++) {
    Acl = Ac + (nC-1)*ldA;
    b0 = Bcl;
    // update all B-values with current A column and current B, AXPY
    for (j = nC; j > 0; j--) {
      Ar = Acl + j - 1;     // diagonal entry on A.

      // update following elements on B row
      _inner_axpy_trans(b0+ldB, Ar+1, b0, alpha, nC-j, ldB);
      // update current element on B rows
      b0[0] = unit ? alpha*b0[0] : alpha*Ar[0]*b0[0];

      b0 -= ldB;        // previous element on B rows
      Acl -= ldA;       // previous column in A
    }
    // next B column, next A column
    Bcl++;
  }
}

//extern void memset(void *, int, size_t);

// X = A*X; unblocked version
void dmmat_trid_unb(mdata_t *B, const mdata_t *A, double alpha, int flags, int N, int S, int E)
{
  // indicates if diagonal entry is unit (=1.0) or non-unit.
  int unit = flags & MTX_UNIT ? 1 : 0;
  double *Bc; 
  
  if (flags & MTX_RIGHT) {
    // for X = X*op(A)
    Bc = &B->md[S];  // row of B
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        _dmmat_trid_axpy_bleft_fwd(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      } else {
        _dmmat_trid_dot_bleft_backwd(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_trid_axpy_bleft_backwd(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      } else {
        _dmmat_trid_dot_bleft_fwd(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      }
    }
  } else {
    // for X = op(A)*X
    Bc = &B->md[S*B->step]; // column of B
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        _dmmat_trid_dot_backward(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      } else {
        _dmmat_trid_axpy_forward(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_trid_dot_forward(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      } else {
        _dmmat_trid_axpy_backward(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      }
    }
  }
  // for X = X*op(A); missing ...
}


// 
void _dmmat_trmm_blk_upper(mdata_t *B, const mdata_t *A,
                           double alpha, int flags, int N, int S, int L, int NB,
                           cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      A0.md = &A->md[i*A->step + i];       // diagonal A block
      A1.md = &A->md[(i+nI)*A->step + i];  // right of diagonal A block
      B0.md = &B->md[j*B->step + i];       // top B block
      B1.md = &B->md[j*B->step + i+nI];    // bottom B block
      
      // update current part with diagonal
      dmmat_trid_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      _dblock_mult_panel(&B0, &A1, &B1, alpha, 0, N-i-nI, nJ, nI, NB, Acpy, Bcpy);
    }
  }
}

void _dmmat_trmm_blk_lower(mdata_t *B, const mdata_t *A,
                           double alpha, int flags, int N, int S, int L, int NB,
                           cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      A0.md = &A->md[i-nI];                  // left of diagonal A block
      A1.md = &A->md[(i-nI)*A->step+(i-nI)]; // diagonal A block
      B0.md = &B->md[j*B->step];             // top B block
      B1.md = &B->md[j*B->step+(i-nI)];      // bottom B block
      
      //printf("i: %d, j: %d, nI: %d, nJ: %d, i-nI: %d\n", i, j, nI, nJ, i-nI);
      //printf("..A1:\n"); print_tile(A1.md, A1.step, nI, nI);
      //printf("..A0:\n"); print_tile(A0.md, A0.step, nI, i-nI);
      //printf("..B1:\n"); print_tile(B1.md, B1.step, nI, nJ);
      //printf("..B0:\n"); print_tile(B0.md, B0.step, i-nI, nJ);
      
      // update current part with diagonal
      dmmat_trid_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      _dblock_mult_panel(&B1, &A0, &B0, alpha, 0, i-nI, nJ, nI, NB, Acpy, Bcpy);
    }
  }
}

// B = B * A; A is upper
void _dmmat_trmm_blk_upper_r(mdata_t *B, const mdata_t *A,
                             double alpha, int flags, int N, int S, int L, int NB,
                             cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      A0.md = &A->md[(i-nI)*A->step];        // above of diagonal A block
      A1.md = &A->md[(i-nI)*A->step+(i-nI)]; // diagonal A block
      B0.md = &B->md[j];                     // right of B block
      B1.md = &B->md[(i-nI)*B->step+j];      // bottom B block
      
      //printf("i: %d, j: %d, nI: %d, nJ: %d, i-nI: %d\n", i, j, nI, nJ, i-nI);
      //printf("..A1:\n"); print_tile(A1.md, A1.step, nI, nI);
      //printf("..A0:\n"); print_tile(A0.md, A0.step, i-nI, nI);
      //printf("..B1:\n"); print_tile(B1.md, B1.step, nJ, nI);
      //printf("..B0:\n"); print_tile(B0.md, B0.step, nJ, i-nI);
      
      // update current part with diagonal
      dmmat_trid_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      _dblock_mult_panel(&B1, &A0, &B0, alpha, 0, i-nI, nJ, nI, NB, Acpy, Bcpy);
    }
  }
}

/*

  A00 | A01 | A02  B0                     A00 | A01 | A02  
  ---------------  --                     ---------------
   0  | A11 | A12  B1         B0|B1|B2     0  | A11 | A12 
  ---------------  --                     ---------------
   0  |  0  | A22  B2                      0  |  0  | A22 

  B0 = A00*B0 + A01*B1 + A02*B2    B0 = B0*A00 
  B1 =          A11*B1 + A12*B2    B1 = B0*A01 + B1*A11
  B2 =                   A22*B2    B2 = B0*A02 + B1*A12 + B2*A22

  B0 = trmm_unb(A00,B0) + [A01; A02] * [B1; B2].T
  B1 = trmm_unb(A11,B1) + A12*B2
  B2 = trmm_unb(A22,B2)
 */
void dmmat_trmm_blk(mdata_t *B, const mdata_t *A, double alpha, int flags,
                    int N, int S, /*int L, int R,*/ int E, int NB)
{
  // S < E <= N
  double Abuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  double Bbuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  cbuf_t Acpy = {Abuf, MAX_VP_COLS*MAX_VP_COLS};
  cbuf_t Bcpy = {Bbuf, MAX_VP_COLS*MAX_VP_COLS};

  if (E-S <= 0)
    return;

  if (NB > MAX_VP_COLS || NB <= 0) {
    NB = MAX_VP_COLS;
  }

  if (flags & MTX_RIGHT) {
    // B = alpha*B*op(A)
    if (flags & MTX_UPPER) {
      _dmmat_trmm_blk_upper_r(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
    } else {
    }

  } else {
    // B = alpha*op(A)*B
    // work it out from left to right
    if (flags & MTX_UPPER) {
      _dmmat_trmm_blk_upper(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
    } else {
      _dmmat_trmm_blk_lower(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
