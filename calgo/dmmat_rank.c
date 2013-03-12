
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

static void
_dmmat_rank_unb_diag(mdata_t *C, const mdata_t *A, double alpha, double beta,
		     int flags,  int P, int S, int E, cbuf_t *Acpy, cbuf_t *Bcpy)
{
}

// SYRK; C = alpha*A*A.T + beta*C
// C is N*N and 0 <= S < E < N; A is N*P and A.T is P*N
void dmmat_rank_unb(mdata_t *C, const mdata_t *A, double alpha, double beta,
		    int flags,  int P, int S, int E)
{
  
  double Abuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  double Bbuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  cbuf_t Acpy = {Abuf, MAX_VP_COLS*MAX_VP_COLS};
  cbuf_t Bcpy = {Bbuf, MAX_VP_COLS*MAX_VP_COLS};

  
}

static void
_dmmat_rank_diag(mdata_t *C, const mdata_t *A, 
		double alpha, double beta,
		int flags,  int P, int nC, int vlen, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  int i, j;
  mdata_t A0 = {A->md, A->step};
  mdata_t B0 = {A->md, A->step};

  if (flags & MTX_UPPER) {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      dscale_vec(C->md, C->step, beta, nC-i);
      // update one row of C  (nC-i columns, 1 row)
      _dblock_mult_panel(C, &A0, &B0, alpha, MTX_TRANSB, P, nC-i, 1, vlen, Acpy, Bcpy);
      // move along the diagonal to next row of C
      C->md += C->step + 1;
      // move A to next row
      A0.md ++;
      // move B to next column
      B0.md ++; //= B0.step;
    }
  } else {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      dscale_vec(C->md, C->step, beta, i+1);
      // update one row of C  (nC-i columns, 1 row)
      _dblock_mult_panel(C, &A0, &B0, alpha, MTX_TRANSB, P, i+1, 1, vlen, Acpy, Bcpy);
      // move to next row of C
      C->md ++;
      // move A to next row
      A0.md ++;
      // move B to next column
      //B0.md ++; // += B0.step;
    }
  }
}

/*
    C00 C01 C02  a0  
     0  C11 C12  a1 * b0 b1 b2
     0   0  C22  a2

    C00 += a0*b0, C01 += a1*b1, C02 += a0*b2
                  C11 += a1*b1, C12 += a1*b2
                                C22 += a2*
 */
void dmmat_rank_blk(mdata_t *C, const mdata_t *A, double alpha, double beta,
		    int flags,  int P, int S, int E, int vlen, int NB)
{
  mdata_t Cd, Ad, Bd;
  double Abuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  double Bbuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  cbuf_t Acpy = {Abuf, MAX_VP_ROWS*MAX_VP_COLS};
  cbuf_t Bcpy = {Bbuf, MAX_VP_ROWS*MAX_VP_COLS};

  register int i, j, nI, nC;
  if (E-S <= 0 || P <= 0) {
    return;
  }
  if (NB > MAX_VP_COLS || NB <= 0) {
    NB = MAX_VP_COLS;
  }
  if (vlen > MAX_VP_ROWS || vlen <= 0) {
    NB = MAX_VP_ROWS;
  }
  if (NB > E-S) {
    NB = E-S;
  }

  Cd.step = C->step;
  Ad.step = A->step;
  Bd.step = A->step;

  printf("S: %d, N: %d, P: %d, NB: %d\n", S, E, P, NB);
  for (i = S; i < E; i += NB) {
    nI = E - i < NB ? E - i : NB;
    
    Cd.md = &C->md[i*C->step+i];
    Ad.md = &A->md[i];
    Bd.md = &A->md[i];
    // 1. update on diagonal
    _dmmat_rank_diag(&Cd, &Ad, alpha, beta, flags, P, nI, vlen, &Acpy, &Bcpy);
    // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
    Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
    Ad.md = &A->md[i];
    Bd.md = flags & MTX_LOWER ? &A->md[S] : &A->md[i+nI];
    nC = flags & MTX_LOWER ? i : E-i-nI;
    _dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, &Acpy, &Bcpy);
  }
}


// SYRK; C = alpha*A*B.T + alpha*B.T*A + beta*C
void dmmat_rank2_unb(mdata_t *C, const mdata_t *A, double alpha, double beta,
		    int flags,  int N, int S, int E)
{
}


// Local Variables:
// indent-tabs-mode: nil
// End: