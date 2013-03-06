
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
     a00 | a01 | a02       b00|b01
     ---------------      --------   
      0  | a11 | a12       b10|b11
     ---------------      --------
      0  |  0  | a22       b20|b21

   Variant 0:
      if A diagonal is non-UNIT
         b10 = (b10 - a10*b00) / a11
      else
         b10 = b10 - a10*b00
   
   Variant 1:
      if A diagonal is non-UNIT
         b10 = b10/a11
         b00 = b00 - a01*b10;
      else
         b10 = b10
         b00 = b00 - a01*b10

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

static void
_dmmat_solve_forward(double *Bc, const double *Ac, double alpha, int flags, 
					 int ldB, int ldA, int nRE, int nB)
{
  int unit = flags & MTX_UNIT ? 1 : 0;
  register int i, j;
  register double *b1, *b2, *Br;
  const double *a11, *a21;

  // lower diagonal matrix of nRE rows/cols and vector X, Y of length nRE
  a11 = Ac;
  Br = Bc;

  //printf("nRE: %d, nB: %d\n", nRE, nB);
  for (i = 0; i < nRE; i++) {
	a11 = Ac + i;                // move on diagonal
	a21 = a11 + 1;
	b1 = Br;
	//printf("..B row: %d\n", i); print_tile(Br, ldB, 1, nB);
	//printf("..A col: %d\n", i); print_tile(a11, ldA, nRE-i, 1);
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

// B = A.-1*B; unblocked
void dmmat_solve_unb(mdata_t *B, const mdata_t *A, double alpha, int flags, int N, int S, int E)
{
  double *Bc;
  //printf("solve_unb: N=%d, S=%d, E=%d\n", N, S, E);
  if (flags & MTX_RIGHT) {
  } else {
    Bc = &B->md[S*B->step];
    if (flags & MTX_LOWER) {
      _dmmat_solve_forward(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
    } else {
      _dmmat_solve_backward(Bc, A->md, alpha, flags, B->step, A->step, N, E-S);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
