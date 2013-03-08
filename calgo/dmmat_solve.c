
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "cmops.h"
#include "inner_axpy.h"
//#include "inner_vec_axpy.h"
#include "inner_ddot.h"
#include "inner_ddot_trans.h"

/*
  A: N*N, UPPER            B: N*2
     a00 | a01 : a02       b00|b01
     ===============      ========   
      0  | a11 : a12       b10|b11
     ---------------      --------
      0  |  0  : a22       b20|b21

    b20 = a22*b'20                       --> b'20 = b20/a22
    b10 = a11*b'10 + a12*b'20            --> b'10 = (b10 - a12*b'20)/a11
    b00 = a00*b'00 + a01*b'10 + a02*b'20 --> b'00 = (b00 - a01*b'10 - a12*b'20)/a00
*/
static void
_dmmat_solve_backward(double *Bc, const double *Ac, double alpha, int flags, 
                      int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  double *b0, *b1, *Bcl;
  const double *a11, *a01, *Acl;
  int unit = flags & MTX_UNIT ? 1 : 0;

  // upper diagonal matrix of nRE rows/cols and matrix B with nRE rows, nB columns
  // move to point to last column of A and B.
  Acl = Ac + (nRE-1)*ldA;
  Bcl = Bc + (nB-1)*ldB;

  for (i = nRE-1; i >= 0; i--) {
    a01 = Acl;
    a11 = a01 + i;  // diagonal entry in A
    b0 = Bcl;
    for (j = 0; j < nB; j++) {
      b1 = b0 + i;
      b1[0] = unit ? b1[0] : b1[0]/a11[0];
      // update all x0-values with in current column (i is the count above current row)
      _inner_daxpy(b0, a01, b1, -1.0, i);

      // repartition: previous column in B
      b0  -= ldB;
    }
    // previous column in A
    Acl -= ldA;
  }
}
/*
  A: N*N, UPPER, TRANS     B: N*2
     a00 : a01 | a02       b00|b01
     ---------------      --------   
      0  : a11 | a12       b10|b11
     ===============      ========
      0  :  0  | a22       b20|b21

   b00 = a00*b'00                       --> b'00 = b00/a00
   b10 = a01*b'00 + a11*b'10            --> b'10 = (b10 - a01*b'00)/a11
   b20 = a02*b'00 + a12*b'10 + a22*b'20 --> b'20 = (b20 - a02*b'00 - a12*b'10)/a22

*/

static void
_dmmat_solve_fwd_trans(double *Bc, const double *Ac, double alpha, int flags, 
                       int ldB, int ldA, int nRE, int nB)
{
  int unit = flags & MTX_UNIT ? 1 : 0;
  register int i, j;
  register double *b1, *b0, *Br;
  double btmp;
  const double *a11, *a01;

  // upper diagonal matrix of nRE rows/cols and B with nRE rows, nB cols
  Br = Bc;

  for (i = 0; i < nRE; i++) {
    a01 = Ac;           // next column in A 
    a11 = a01 + i;      // move on diagonal
    b0 = Bc;            // b0 is start of column
    for (j = 0; j < nB; j++) {
      b1 = b0 + i;
      btmp = 0.0;
      // update current element with b0-values
      _inner_ddot(&btmp, a01, b0, 1.0, i);
      b1[0] = unit ? b1[0] - btmp : (b1[0] - btmp)/a11[0];

      // next column
      b0 += ldB;
    }
    // next column in A 
    Ac += ldA;
  }
}

/*
  A: N*N, LOWER            B: N*2
     a00 |  0  :  0        b00|b01
     ===============      ========   
     a10 | a11 :  0        b10|b11
     ---------------      --------
     a20 | a21 : a22       b20|b21

    b00 = a00*b'00                       --> b'00 = b020/a00
    b10 = a10*b'00 + a11*b'10            --> b'10 = (b10 - a10*b'00)/a11
    b20 = a20*b'00 + a21*b'10 + a22*b'20 --> b'20 = (b20 - a20*b'00 - a21*b'10)/a22
 */
static void
_dmmat_solve_forward(double *Bc, const double *Ac, double alpha, int flags, 
                     int ldB, int ldA, int nRE, int nB)
{
  int unit = flags & MTX_UNIT ? 1 : 0;
  register int i, j;
  register double *b1, *b2, *Br;
  const double *a11, *a21;

  // A is lower diagonal matrix of nRE rows/cols and matrix B is nRE rows, nB cols
  a11 = Ac;
  Br = Bc;

  for (i = 0; i < nRE; i++) {
    a11 = Ac + i;                // move on diagonal
    a21 = a11 + 1;
    b1 = Br;
    for (j = 0; j < nB; j++) {
      b2 = b1 + 1;
      b1[0] = unit ? b1[0] : b1[0]/a11[0];
      // update all b2-values with in current column
      _inner_daxpy(b2, a21, b1, -1.0, nRE-1-i);
      b1 += ldB;
    }
    // next B row, next column in A 
    Br ++;
    Ac += ldA;
  }
}

/*
  A: N*N, LOWER, TRANS     B: N*2
     a00 |  0  :  0        b00|b01
     ===============      ========   
     a10 | a11 :  0        b10|b11
     ---------------      --------
     a20 | a21 : a22       b20|b21

    b00 = a00*b'00 + a10*b'10 + a20*b'20 --> b'00 = (b00 - a10*b'10 - a20*b'20)/a00
    b10 = a11*b'10 + a21*b'20            --> b'10 = (b10 - a21*b'20)/a11
    b20 = a22*b'20                       --> b'20 = b20/a22
 */
static void
_dmmat_solve_backwd_trans(double *Bc, const double *Ac, double alpha, int flags, 
                           int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  register double *b1, *b2, *Bcl;
  register const double *a11, *a21, *Acl;
  double btmp;
  int unit = flags & MTX_UNIT ? 1 : 0;

  // upper diagonal matrix of nRE rows/cols and matrix B with nRE rows, nB columns
  // move to point to last column of A and B.
  Acl = Ac + (nRE-1)*ldA;
  Bcl = Bc + (nB-1)*ldB;

  for (i = nRE-1; i >= 0; i--) {
    a11 = Acl + i;  // diagonal entry in A
    a21 = a11 + 1;
    b1 = Bcl + i;
    for (j = 0; j < nB; j++) {
      b2 = b1 + 1;
      // update current value with previous values.
      btmp = 0.0;
      _inner_ddot(&btmp, a21, b2, 1.0, nRE-1-i);
      b1[0] = unit ? b1[0] - btmp : (b1[0] - btmp)/a11[0];

      // repartition: previous column in B
      b1  -= ldB;
    }
    // previous column in A
    Acl -= ldA;
  }
}

/*
    B: 2*N        A: N*N, UPPER
                  a00 | a01 : a02  
    b00|b01|b02   ===============  
    -----------    0  | a11 : a12  
    b10|b11|b12   ---------------  
                   0  |  0  : a22  

    b00 = a00*b'00                       --> b'00 = b00/a00
    b01 = a01*b'00 + a11*b'01            --> b'01 = (b01 - a01*b'00)/a11
    b02 = a02*b'00 + a12*b'01 + a22*b'02 --> b'02 = (b02 - a02*b'00 - a12*b'01)/a22

*/
static void
_dmmat_solve_bleft_fwd(double *Bc, const double *Ac, double alpha, int flags, 
                       int ldB, int ldA, int nRE, int nB)
{
  int unit = flags & MTX_UNIT ? 1 : 0;
  register int i, j;
  register double *b0, *b1, *Bcl;
  register const double *a11, *Acl;
  double btmp;

  // A is lower diagonal matrix of nRE rows/cols, B is nB rows, nRE cols
  Bcl = Bc;

  for (i = 0; i < nB; i++) {
    b0 = Bcl;
    b1 = b0;
    Acl = Ac;
    for (j = 0; j < nRE; j++) {
      a11 = Acl + j;    // diagonal entry
      btmp = 0.0;
      _inner_ddot_trans(&btmp, Acl, b0, 1.0, j, ldB);
      // update current value with previous values.
      b1[0] = unit ? b1[0] - btmp : (b1[0] - btmp)/a11[0];
      b1 += ldB;
      Acl += ldA;
    }
    // next B row
    Bcl++;
  }
}
/*
    B: 2*N        A: N*N, UPPER, TRANS
                  a00 | a01 : a02  
    b00|b01|b02   ===============  
    -----------    0  | a11 : a12  
    b10|b11|b12   ---------------  
                   0  |  0  : a22  

    b00 = a00*b'00 + a01*b'01 + a02*b'02 --> b'00 = (b00 - a01*b'01 - a02*b'02)/a00
    b01 = a11*b'01 + a12*b'02            --> b'01 = (b01            - a12*b'02)/a11
    b02 = a22*b'02                       --> b'02 = b02/a22

*/
static void
_dmmat_solve_bleft_backwd_trans(double *Bc, const double *Ac, double alpha, int flags, 
                                int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  register double *b1, *b0, *Bcl;
  register const double *a11, *a21, *Acl;
  double btmp;
  int unit = flags & MTX_UNIT ? 1 : 0;

  // A is lower diagonal matrix of nRE rows/cols and matrix B with nB rows, nRE cols
  // move to point to last column of A and B.
  Bcl = Bc + (nRE-1)*ldB;
  b0 = Bc;
  for (i = 0; i < nB; i++) {
    Acl = Ac + (nRE-1)*ldA;
    b1 = Bcl;
    for (j = nRE-1; j >= 0; j--) {
      a11 = Acl + j;  // diagonal entry in A
      b1[0] = unit ? b1[0] : b1[0]/a11[0];
      // update preceeding values with current values
      _inner_axpy_trans(b0, Acl, b1, -1.0, j, ldB);

      // repartition: previous column in B, A
      b1  -= ldB;
      Acl -= ldA;
    }
    // next row in B
    Bcl++;
    b0++;
  }
}

/*
    B: 2*N        A: N*N, LOWER
                  a00 |  0  :  0  
    b00|b01|b02   ===============  
    -----------   a10 | a11 :  0  
    b10|b11|b12   ---------------  
                  a20 | a21 : a22  

    b00 = a00*b'00 + a10*b'01 + a20*b'02 --> b'00 = (b00 - a10*b'01 - a20*b'02)/a00
    b01 = a11*b'01 + a21*b'02            --> b'01 = (b01            - a21*b'02)/a11
    b02 = a22*b'02                       --> b'02 = b02/a22

*/
static void
_dmmat_solve_bleft_backwd(double *Bc, const double *Ac, double alpha, int flags, 
                          int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  register double *b1, *b2, *Bcl;
  register const double *a11, *a21, *Acl;
  double btmp;
  int unit = flags & MTX_UNIT ? 1 : 0;

  // A is lower diagonal matrix of nRE rows/cols and matrix B with nB rows, nRE cols
  // move to point to last column of A and B.
  Bcl = Bc + (nRE-1)*ldB;
  
  for (i = 0; i < nB; i++) {
    Acl = Ac + (nRE-1)*ldA;
    b1 = Bcl;
    b2 = b1;
    for (j = nRE-1; j >= 0; j--) {
      a11 = Acl + j;  // diagonal entry in A
      a21 = a11 + 1;
      // update current value with previous values.
      btmp = 0.0;
      _inner_ddot_trans(&btmp, a21, b2, 1.0, nRE-1-j, ldB);
      b1[0] = unit ? b1[0] - btmp : (b1[0] - btmp)/a11[0];

      // repartition: previous column in B, A
      b2 = b1;
      b1  -= ldB;
      Acl -= ldA;
    }
    // next row in B
    Bcl++;
  }
}

/*
    B: 2*N        A: N*N, LOWER, TRANS
                  a00 |  0  :  0  
    b00|b01|b02   ===============  
    -----------   a10 | a11 :  0  
    b10|b11|b12   ---------------  
                  a20 | a21 : a22  

    b00 = a00*b'00                       --> b'00 = b00/a00
    b01 = a10*b'00 + a11*b'01            --> b'01 = (b01 - a10*b'00)/a11
    b02 = a20*b'00 + a21*b'01 + a22*b'02 --> b'02 = (b02 - a20*b'00 - a21*b'01)/a22

*/
static void
_dmmat_solve_bleft_fwd_trans(double *Bc, const double *Ac, double alpha, int flags, 
                             int ldB, int ldA, int nRE, int nB)
{
  register int i, j;
  register double *b1, *b2, *Bcl;
  register const double *a11, *a21, *Acl;
  double btmp;
  int unit = flags & MTX_UNIT ? 1 : 0;

  // A is lower diagonal matrix of nRE rows/cols and matrix B with nB rows, nRE cols
  // move to point to last column of A and B.
  Bcl = Bc;
  for (i = 0; i < nB; i++) {
    Acl = Ac;
    b1 = Bcl;
    for (j = 0; j < nRE; j++) {
      a11 = Acl + j;  // diagonal entry in A
      b2 = b1 + ldB;
      b1[0] = unit ? b1[0] : b1[0]/a11[0];

      // update following values with current values
      _inner_axpy_trans(b2, Acl, b1, -1.0, nRE-1-j, ldB);

      // repartition: previous column in B, A
      b1  += ldB;
      Acl += ldA;
    }
    // next row in B
    Bcl++;
  }
}

// B = A.-1*B; unblocked
void dmmat_solve_unb(mdata_t *B, const mdata_t *A, double alpha, int flags, int N, int S, int E)
{
  double *Bc;
  //printf("solve_unb: N=%d, S=%d, E=%d\n", N, S, E);
  if (flags & MTX_RIGHT) {
    // for B = alpha*B*op(A)
    Bc = &B->md[S]; 
    if (flags & MTX_LOWER) {
      if (flags & MTX_TRANSA) {
        _dmmat_solve_bleft_fwd_trans(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      } else {
        _dmmat_solve_bleft_backwd(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_solve_bleft_backwd_trans(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      } else {
        _dmmat_solve_bleft_fwd(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      }
    }
  } else {
    Bc = &B->md[S*B->step];
    if (flags & MTX_LOWER) {
      if (flags & MTX_TRANSA) {
        _dmmat_solve_backwd_trans(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      } else {
        _dmmat_solve_forward(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_solve_fwd_trans(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      } else {
        _dmmat_solve_backward(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
      }
    }
  }
}

/*  UPPER, LEFT

    A00 | A01 | A02   B0
   ----------------   --
     0  | A11 | A12   B1
   ----------------   --
     0  |  0  | A22   B2

    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = A00.-1*(B0 - A01*B'1 - A02*B'2)
    B1 = A11*B'1 + A12*B'2           --> B'1 = A11.-1*(B1           - A12*B'2)
    B2 = A22*B'2                     --> B'2 = A22.-1*B2
 */
static void
_dmmat_solve_blk_upper(mdata_t *B, const mdata_t *A, double alpha, int flags,
                       int N, int S, int E, int NB, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB;
    for (j = E; j > S; j -= NB) {
      nJ = j < S + NB ? S + NB - j : NB;
      cJ = j < S + NB ? S : j-NB;

      A0.md = &A->md[cI*A->step];           // above the diagonal A block
      A1.md = &A->md[cI*A->step + cI];      // diagonal A block
      B0.md = &B->md[cJ*B->step ];          // top B block
      B1.md = &B->md[cJ*B->step + cI];      // bottom B block

      // solve bottom block
      dmmat_solve_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      // update top with bottom solution
      _dblock_mult_panel(&B0, &A0, &B1, -1.0, 0, nI, nJ, i-nI, NB, Acpy, Bcpy);
    }
  }
}

/*  UPPER, TRANS, LEFT

    A00 | A01 | A02   B0
   ----------------   --
     0  | A11 | A12   B1
   ----------------   --
     0  |  0  | A22   B2

    B0 = A00*B'0                     --> B'0 = A00.-1*B0
    B1 = A01*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A01*B'0)
    B2 = A02*B'0 + A12*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A02*B'0 - A12*B'1)
 */
static void
_dmmat_solve_blk_u_trans(mdata_t *B, const mdata_t *A, double alpha, int flags,
                         int N, int S, int E, int NB, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;
    for (j = S; j < E; j += NB) {
      nJ = j < E - NB ? NB : E - j;
      cJ = nJ < NB ? E - nJ : j;

      A0.md = &A->md[cI*A->step];        // above the diagonal A block
      A1.md = &A->md[cI*A->step + cI];   // diagonal A block
      B0.md = &B->md[cJ*B->step];        // top B block
      B1.md = &B->md[cJ*B->step + cI];   // bottom B block

      printf("i:%d, j:%d, nI:%d, nJ:%d, N-i-nI:%d, cI:%d, cJ:%d\n", i, j, nI, nJ, N-i-nI, cI, cJ);
      printf("..A0:\n"); print_tile(A0.md, A0.step, cI, nI);
      printf("..B0:\n"); print_tile(B0.md, B0.step, cI, nJ);
      printf("..A1:\n"); print_tile(A1.md, A1.step, nI, nI);
      printf("..B1:\n"); print_tile(B1.md, B1.step, nI, nJ);

      // update bottom block with top block
      _dblock_mult_panel(&B1, &A0, &B0, -1.0, MTX_TRANSA, nI, nJ, cI, NB, Acpy, Bcpy);
      printf("=1 B1:\n"); print_tile(B1.md, B1.step, nI, nJ);
      printf("   B0:\n"); print_tile(B0.md, B0.step, cI, nJ);
      // solve bottom block
      dmmat_solve_unb(&B1, &A1, alpha, flags, nI, 0, nJ);
      printf("=2 B1:\n"); print_tile(B1.md, B1.step, nI, nJ);
      printf("   B0:\n"); print_tile(B0.md, B0.step, cI, nJ);
    }
  }
}

/*  LOWER, LEFT

    A00 |  0  |  0    B0
   ----------------   --
    A10 | A11 |  0    B1
   ----------------   --
    A20 | A21 | A22   B2

    B0 = A00*B'0                     --> B'0 = A00.-1*B0
    B1 = A10*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A10*B'0)
    B2 = A20*B'0 + A21*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A20*B'0 - A21*B'1)
 */
static void
_dmmat_solve_blk_lower(mdata_t *B, const mdata_t *A, double alpha, int flags,
                       int N, int S, int E, int NB, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = 0; i < N; i += NB) {
    nI = i < N - NB ? NB : N - i;
    cI = nI < NB ? N-nI : i;
    for (j = S; j < E; j += NB) {
      nJ = j < E - NB ? NB : E - j;
      cJ = nJ < NB ? E - nJ : j;

      A0.md = &A->md[cI*A->step + cI];      // diagonal A block
      A1.md = &A->md[cI*A->step + cI+nI];   // below the diagonal A block
      B0.md = &B->md[cJ*B->step + cI];      // top B block
      B1.md = &B->md[cJ*B->step + cI+nI];   // bottom B block

      // solve top block
      dmmat_solve_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
      // update bottom block with top block
      _dblock_mult_panel(&B1, &A1, &B0, -1.0, 0, nI, nJ, N-i-nI, NB, Acpy, Bcpy);
    }
  }
}

/*  LOWER, TRANSA, LEFT

    A00 |  0  |  0    B0
   ----------------   --
    A10 | A11 |  0    B1
   ----------------   --
    A20 | A21 | A22   B2

    B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = A00.-1*(B0 - A10*B'1 - A20*B'2)
    B1 = A11*B'1 + A21*B'2           --> B'1 = A11.-1*(B1           - A21*B'2)
    B2 = A22*B'2                     --> B'2 = A22.-1*B2
 */
static void
_dmmat_solve_blk_l_trans(mdata_t *B, const mdata_t *A, double alpha, int flags,
                         int N, int S, int E, int NB, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, j, nI, nJ, cI, cJ;
  mdata_t A0, A1, B0, B1;
  A0.step = A->step;
  A1.step = A->step;
  B0.step = B->step;
  B1.step = B->step;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    cI = i < NB ? 0 : i-NB;
    for (j = E; j > S; j -= NB) {
      nJ = j < S + NB ? S + NB - j : NB;
      cJ = j < S + NB ? S : j-NB;

      A0.md = &A->md[cI*A->step + cI];      // diagonal A block
      A1.md = &A->md[cI*A->step + i];        // below the diagonal A block
      B0.md = &B->md[cJ*B->step + cI];      // top B block
      B1.md = &B->md[cJ*B->step + i];       // bottom B block

      //printf("i:%d, j:%d, nI:%d, nJ:%d, N-i:%d, cI:%d, cJ:%d\n", i, j, nI, nJ, N-i, cI, cJ);
      //printf("..A0:\n"); print_tile(A0.md, A0.step, nI, nI);
      //printf("..B0:\n"); print_tile(B0.md, B0.step, nI, nJ);
      //printf("..A1:\n"); print_tile(A1.md, A1.step, N-i, nI);
      //printf("..B1:\n"); print_tile(B1.md, B1.step, N-i, nJ);

      // update top with bottom solution
      _dblock_mult_panel(&B0, &A1, &B1, -1.0, MTX_TRANSA, N-i, nJ, nI, NB, Acpy, Bcpy);
      // solve top block
      dmmat_solve_unb(&B0, &A0, alpha, flags, nI, 0, nJ);
    }
  }
}

// B = A.-1*B; unblocked
void dmmat_solve_blk(mdata_t *B, const mdata_t *A, double alpha, int flags,
                     int N, int S, int E, int NB)
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
  } else {
    // B = A.-1*B; B = A.-T*B
    if (flags & MTX_UPPER) {
      if (flags & MTX_TRANSA) {
        _dmmat_solve_blk_u_trans(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
      } else {
        _dmmat_solve_blk_upper(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
      }
    } else {
      if (flags & MTX_TRANSA) {
        _dmmat_solve_blk_l_trans(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
      } else {
        _dmmat_solve_blk_lower(B, A, alpha, flags, N, S, E, NB, &Acpy, &Bcpy);
      }
    }
  }
}
// Local Variables:
// indent-tabs-mode: nil
// End:
