
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

// Local Variables:
// indent-tabs-mode: nil
// End:
