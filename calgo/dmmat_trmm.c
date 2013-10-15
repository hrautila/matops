
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "mvec_nosimd.h"

extern
void __dmult_inner_a_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              double alpha, int flags,
                              int P, int nSL, int nR, 
                              int KB, int NB, int MB, mdata_t *Acpy, mdata_t *Bcpy);



// Functions here implement various versions of TRMM operation.

/*
  B = A*B; A is lower

    a00| 0 | 0   b00
    a10|a11| 0   b10
    a20|a21|a22  b20

  b00 = a00*b00
  b10 = a10*b00 + a11*b10
  b20 = a20*b00 + a21*b10 + a22*b20

  --> work it backwards as b20 is not needed for b10, ...

  Calculates backward a diagonal block and updates Xc values from last to first.
  Updates are calculated in breadth first manner by with successive AXPY operations.
 */
static void
_dmmat_trmm_unb_lower(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1axpy(b0+1, 1, Ar+1, b0, 1, alpha, nRE-i);
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

    a00|a01|a02  b00
     0 |a11|a12  b10
     0 | 0 |a22  b20

  b00 = a00*b00
  b10 = a01*b00 + a11*b10
  b20 = a02*b00 + a12*b10 + a22*b20

  --> work it backwards with DOT products

  Calculates backward a diagonal block and updates Xc values from last to first.
  Updates are calculated in depth first manner with DOT operations.
 */
static void
_dmmat_trmm_unb_u_trans(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1dot(&xtmp, 1, Acl, b0, 1, alpha, nRE-unit-i);
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

      a00|a01|a02  b00
       0 |a11|a12  b10
       0 | 0 |a22  b20

    b00 = a00*b00 + a01*b10 + a02*b20
    b10 =           a11*b10 + a12*b20
    b20 =                     a22*b20

    --> work it forwards as b00, b01 not need for later elements; AXPY

 Calculate forward a diagonal block and updates Xc values from first to last.
*/

static void
_dmmat_trmm_unb_upper(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1axpy(Bc, 1, Ac, b0, 1, alpha, i);
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
_dmmat_trmm_unb_l_trans(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1dot(&xtmp, 1, Ar, b0+unit, 1, alpha, nRE-unit-i);
      b0[0] = xtmp;
      b0 += ldB;
    }
    // next row in B, next column in A 
    Bcr++;
    Ac += ldA;
  }
}

/* UPPER, RIGHT
  for B = B*A; A is [nC, nC], B is [nRE, nC]
  
                 a00|a01|a02
    b00|b01|b02   0 |a11|a12
                  0 | 0 |a22

    b00 = b00*a00
    b01 = b00*a01 + a11*b01
    b02 = b00*a02 + a12*b01 + a22*b02
    
    --> work it backwards as b12 & b02 are not needed for b11, b01, ...
*/
static void
_dmmat_trmm_unb_r_upper(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1dot(&btmp, 1, Ar, Br, ldB, alpha, j-unit);
      b0[0] = btmp;
      b0++;
      Br++;
    }
    // previous B column, previous A column
    Bcl -= ldB;
    Acl -= ldA;
  }
}

/* LOWER, RIGHT,
  for B = B*A; A is [nC, nC], B is [nRE, nC], 
  
                 a00| 0 | 0
    b00|b01|b02  a10|a11| 0
                 a20|a21|a22

    b00 = b00*a00 + b01*a10 + b02*a20
    b01 = b01*a11 + a21*b02
    b02 = b02*a22
    
    --> work it forward as b00 are not needed for b01, b02, ... with DOT
*/
static void
_dmmat_trmm_unb_r_lower(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1dot(&btmp, 1, Ar+unit, Br+unit*ldB, ldB, alpha, j-unit);
      b0[0] = unit ? btmp + alpha*b0[0] : btmp;
      b0++;
      Br++;
    }
    // next B column, next A column
    Bcl += ldB;
    Acl += ldA;
  }
}

/* UPPER, RIGHT, TRANS
  for B = B*A.T; A.T is [nC, nC], B is [nRE, nC], 
  
                 a00|a01|a02
    b00|b01|b02   0 |a11|a12
                  0 | 0 |a22

    b00 = b00*a00 + a01*b01 + a02*b02
    b01 =           a11*b01 + a12*b02
    b02 =                     a22*b02
    
    --> work it forward as b00 & b010 are not needed for b01, b11, ... with AXPY
*/
static void
_dmmat_trmm_unb_ru_trans(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1axpy(Bcl, ldB, Acl, b0, ldB, alpha, j);
      // update current element on B rows
      b0[0] = unit ? alpha*b0[0] : alpha*Ar[0]*b0[0];
      b0 += ldB;        // next element on B rows
      Acl += ldA;       // next column in A
    }
    // next B column, next A column
    Bcl++;
  }
}

/* LOWER, RIGHT, TRANSA
  for B = B*A.T; A.T is [nC, nC], B is [nRE, nC], 
  
                 a00| 0 | 0
    b00|b01|b02  a10|a11| 0
                 a20|a21|a22

    b00 = b00*a00
    b01 = b00*a10 + a11*b01
    b02 = b00*a20 + a21*b01 + a22*b02
    
    --> work it backward as b02 & b12 are not needed for b01, b11, ... with AXPY
*/
static void
_dmmat_trmm_unb_rl_trans(double *Bc, const double *Ac, double alpha, int unit,
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
      __vmult1axpy(b0+ldB, ldB, Ar+1, b0, ldB, alpha, nC-j);
      // update current element on B rows
      b0[0] = unit ? alpha*b0[0] : alpha*Ar[0]*b0[0];

      b0 -= ldB;        // previous element on B rows
      Acl -= ldA;       // previous column in A
    }
    // next B row
    Bcl++;
  }
}

// X = A*X; unblocked version
void dmmat_trmm_unb(mdata_t *B, const mdata_t *A, double alpha, int flags, int N, int S, int E)
{
  // indicates if diagonal entry is unit (=1.0) or non-unit.
  int unit = flags & MTX_UNIT ? 1 : 0;
  double *Bc; 
  
  if (flags & MTX_RIGHT) {
    // for X = X*op(A)
    Bc = &B->md[S];  // row of B
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        // axpy_left_fwd
        _dmmat_trmm_unb_ru_trans(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      } else {
        // dot_bleft_backward
        _dmmat_trmm_unb_r_upper(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      }
    } else {
      if (flags & MTX_TRANSA) {
        // axpy_bleft_backwd
        _dmmat_trmm_unb_rl_trans(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      } else {
        // dot_bleft_fwd
        _dmmat_trmm_unb_r_lower(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      }
    }
  } else {
    // for X = op(A)*X
    Bc = &B->md[S*B->step]; // column of B
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        // dot_backward
        _dmmat_trmm_unb_u_trans(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      } else {
        // axpy_forward
        _dmmat_trmm_unb_upper(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      }
    } else {
      if (flags & MTX_TRANSA) {
        // dot_forward
        _dmmat_trmm_unb_l_trans(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      } else {
        // axpy_backward
        _dmmat_trmm_unb_lower(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      }
    }
  }
  // for X = X*op(A); missing ...
}


/*  UPPER, LEFT

    A00 | A01 | A02   B0
   ----------------   --
     0  | A11 | A12   B1
   ----------------   --
     0  |  0  | A22   B2

    B0 = A00*B0 + A01*B1 + A02*B2
    B1 = A11*B1 + A12*B2         
    B2 = A22*B2                  
 */
void _dmmat_trmm_blk_upper(mdata_t *B, const mdata_t *A,
                           double alpha, int flags, int N, int S, int L,
                           int KB, int NB, int MB,
                           mdata_t *Acpy, mdata_t *Bcpy)
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
      dmmat_trmm_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B0, &A1, &B1, alpha, 0, N-i-nI, nJ, nI,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}
/*  UPPER, LEFT, TRANSA

    A00 | A01 | A02   B0
   ----------------   --
     0  | A11 | A12   B1
   ----------------   --
     0  |  0  | A22   B2

    B0 = A00*B0
    B1 = A01*B0 + A11*B1
    B2 = A02*B0 + A12*B1 + A22*B2                  
*/
void _dmmat_trmm_blk_u_trans(mdata_t *B, const mdata_t *A,
                             double alpha, int flags, int N, int S, int L,
                             int KB, int NB, int MB,
                             mdata_t *Acpy, mdata_t *Bcpy)
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

      A0.md = &A->md[(i-nI)*A->step];        // above the diagonal A block
      A1.md = &A->md[(i-nI)*A->step+(i-nI)]; // diagonal A block
      B0.md = &B->md[j*B->step];             // top B block
      B1.md = &B->md[j*B->step+(i-nI)];      // bottom B block
      
      // update current part with diagonal
      dmmat_trmm_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B1, &A0, &B0, alpha, MTX_TRANSA, i-nI, nJ, nI,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}


/*  LOWER, LEFT

    A00 |  0  |  0    B0
   ----------------   --
    A10 | A11 |  0    B1
   ----------------   --
    A20 | A21 | A22   B2

    B0 = A00*B0                  
    B1 = A10*B0 + A11*B1         
    B2 = A20*B0 + A21*B1 + A22*B2
 */
void _dmmat_trmm_blk_lower(mdata_t *B, const mdata_t *A,
                           double alpha, int flags, int N, int S, int L,
                           int KB, int NB, int MB,
                           mdata_t *Acpy, mdata_t *Bcpy)
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
      A1.md = &A->md[(i-nI)*A->step+(i-nI)]; // current A block, diagonal
      B0.md = &B->md[j*B->step];             // top B block
      B1.md = &B->md[j*B->step+(i-nI)];      // current B block
      
      // update current part with diagonal
      dmmat_trmm_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B1, &A0, &B0, alpha, 0, i-nI, nJ, nI,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}
/*  LOWER, LEFT, TRANSA

    A00 |  0  |  0    B0
   ----------------   --
    A10 | A11 |  0    B1
   ----------------   --
    A20 | A21 | A22   B2

    B0 = A00*B0 + A10*B1 + A20*B2
    B1 = A11*B1 + A21*B2
    B2 = A22*B2
 */
void _dmmat_trmm_blk_l_trans(mdata_t *B, const mdata_t *A,
                             double alpha, int flags, int N, int S, int L,
                             int KB, int NB, int MB,
                             mdata_t *Acpy, mdata_t *Bcpy)
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

      A0.md = &A->md[i*A->step + i];       // current A block, diagonal
      A1.md = &A->md[i*A->step + i+nI];    // below of diagonal A block
      B0.md = &B->md[j*B->step + i];       // current B block, top
      B1.md = &B->md[j*B->step + i+nI];    // bottom B block
      
      // update current part with diagonal
      dmmat_trmm_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B0, &A1, &B1, alpha, MTX_TRANSA, N-i-nI, nJ, nI,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}

/*  UPPER, RIGHT

                 A00 | A01 | A02 
                ---------------- 
    B0|B1|B2      0  | A11 | A12 
                ---------------- 
                  0  |  0  | A22 

    B0 = B0*A00                   = trmm_unb(B0, A00)
    B1 = B0*A01 + B1*A11          = trmm_unb(B1, A11) + B0*A01
    B2 = B0*A02 + B1*A12 + B2*A22 = trmm_unb(B2, A22) + [B0; B1]*[A02; A12].T
 */
void _dmmat_trmm_blk_r_upper(mdata_t *B, const mdata_t *A,
                             double alpha, int flags, int N, int S, int L,
                             int KB, int NB, int MB,
                             mdata_t *Acpy, mdata_t *Bcpy)
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
      A1.md = &A->md[(i-nI)*A->step+(i-nI)]; // current A, diagonal block
      B0.md = &B->md[j];                     // left of B block
      B1.md = &B->md[(i-nI)*B->step+j];      // current B block
      
      // update current part with diagonal
      dmmat_trmm_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B1, &B0, &A0, alpha, 0, i-nI, nI, nJ,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}

/*  UPPER, RIGHT, TRANS

                 A00 | A01 | A02 
                ---------------- 
    B0|B1|B2      0  | A11 | A12 
                ---------------- 
                  0  |  0  | A22 

    B0 = B0*A00 + B1*A01 + B2*A02 = trmm_unb(B0, A00) + [B1; B2]*[A01; A02].T
    B1 = B1*A11 + B2*A12          = trmm_unb(B1, A11) + B2*A12
    B2 = B2*A22                   = trmm_unb(B2, A22)
 */
void _dmmat_trmm_blk_ru_trans(mdata_t *B, const mdata_t *A,
                              double alpha, int flags, int N, int S, int L,
                              int KB, int NB, int MB,
                              mdata_t *Acpy, mdata_t *Bcpy)
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
      B0.md = &B->md[i*B->step + j];       // left B block
      B1.md = &B->md[(i+nI)*B->step + j];  // right B block
      
      // update current part with diagonal; left with diagonal
      dmmat_trmm_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B0, &B1, &A1, alpha, MTX_TRANSB, N-i-nI, nI, nJ,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}

/*  LOWER, RIGHT

                 A00 |  0  |  0
                ---------------- 
    B0|B1|B2     A01 | A11 |  0
                ---------------- 
                 A02 | A12 | A22 

    B0 = B0*A00 + B1*A01 + B2*A02
    B1 = B1*A11 + B2*A12
    B2 = B2*A22
 */
void _dmmat_trmm_blk_r_lower(mdata_t *B, const mdata_t *A,
                             double alpha, int flags, int N, int S, int L,
                             int KB, int NB, int MB,
                             mdata_t *Acpy, mdata_t *Bcpy)
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
      A1.md = &A->md[i*A->step + i+nI];    // below the diagonal A block
      B0.md = &B->md[i*B->step + j];       // current B block, left
      B1.md = &B->md[(i+nI)*B->step + j];  // right B block
      
      // update current part with diagonal; left with diagonal
      dmmat_trmm_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B0, &B1, &A1, alpha, 0, N-i-nI, nI, nJ,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}

/*  LOWER, RIGHT, TRANSA

                 A00 |  0  |  0
                ---------------- 
    B0|B1|B2     A01 | A11 |  0
                ---------------- 
                 A02 | A12 | A22 

    B0 = B0*A00
    B1 = B0*A01 + B1*A11
    B2 = B0*A02 + B1*A12 + B2*A22
 */
void _dmmat_trmm_blk_rl_trans(mdata_t *B, const mdata_t *A,
                              double alpha, int flags, int N, int S, int L,
                              int KB, int NB, int MB,
                              mdata_t *Acpy, mdata_t *Bcpy)
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
      A1.md = &A->md[(i-nI)*A->step+(i-nI)]; // current A, diagonal block
      B0.md = &B->md[j];                     // left of B block
      B1.md = &B->md[(i-nI)*B->step+j];      // current B block
      
      // update current part with diagonal
      dmmat_trmm_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update current part with rest of the A, B panels
      __dmult_inner_a_no_scale(&B1, &B0, &A0, alpha, MTX_TRANSB, i-nI, nI, nJ,
                               KB, NB, MB, Acpy, Bcpy);
    }
  }
}


void dmmat_trmm_blk(mdata_t *B, const mdata_t *A, double alpha, int flags,
                    int N, int S, int E, int KB, int NB, int MB)
{
  // S < E <= N
  mdata_t Acpy, Bcpy;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Acpy.md = Abuf;
  Acpy.step = MAX_KB;
  Bcpy.md = Bbuf;
  Bcpy.step = MAX_KB;

  if (E-S <= 0 || N <= 0)
    return;

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  if (flags & MTX_RIGHT) {
    // B = alpha*B*op(A)
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        _dmmat_trmm_blk_ru_trans(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      } else {
        _dmmat_trmm_blk_r_upper(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_trmm_blk_rl_trans(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      } else {
        _dmmat_trmm_blk_r_lower(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      }
    }

  } else {
    // B = alpha*op(A)*B
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        _dmmat_trmm_blk_u_trans(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      } else {
        _dmmat_trmm_blk_upper(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_trmm_blk_l_trans(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      } else {
        _dmmat_trmm_blk_lower(B, A, alpha, flags, N, S, E, KB, NB, MB, &Acpy, &Bcpy);
      }
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
