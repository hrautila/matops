
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

extern
void __dmult_inner_a_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              double alpha, int flags,
                              int P, int nSL, int nRE, 
                              int KB, int NB, int MB, mdata_t *Acpy, mdata_t *Bcpy);

static void
_dmmat_rank_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                 double alpha, double beta,
                 int flags,  int P, int nC,
                 int KB, int NB, int MB,
                 mdata_t *Acpy, mdata_t *Bcpy)
                 //int vlen, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, incA, trans;
  mdata_t A0 = {A->md, A->step};
  mdata_t B0 = {B->md, B->step};

  incA = flags & MTX_TRANSA ? A->step : 1;
  trans = flags & MTX_TRANSA ? MTX_TRANSA : MTX_TRANSB;

  if (flags & MTX_UPPER) {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      dscale_vec(C->md, C->step, beta, nC-i);
      // update one row of C  (nC-i columns, 1 row)
      //_dblock_mult_panel(C, &A0, &B0, alpha, trans, P, nC-i, 1, vlen, Acpy, Bcpy);
      __dmult_inner_a_no_scale(C, &A0, &B0, alpha, trans, P, nC-i, 1, KB, NB, MB, Acpy, Bcpy);
      // move along the diagonal to next row of C
      C->md += C->step + 1;
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incA; 
    }
  } else {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      dscale_vec(C->md, C->step, beta, i+1);
      // update one row of C  (nC-i columns, 1 row)
      //_dblock_mult_panel(C, &A0, &B0, alpha, trans, P, i+1, 1, vlen, Acpy, Bcpy);
      __dmult_inner_a_no_scale(C, &A0, &B0, alpha, trans, P, i+1, 1, KB, NB, MB, Acpy, Bcpy);
      // move to next row of C
      C->md ++;
      // move A to next row
      A0.md += incA;
    }
  }
}

/*
    C00 C01 C02  a0  
     0  C11 C12  a1 * b0 b1 b2
     0   0  C22  a2

 */
void dmmat_rank_blk(mdata_t *C, const mdata_t *A, double alpha, double beta,
		    int flags,  int P, int S, int E, int KB, int NB, int MB)
{
  register int i, j, nI, nC;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Acpy.md = Abuf;
  Acpy.step = MAX_KB;
  Bcpy.md = Bbuf;
  Bcpy.step = MAX_KB;

  if (E-S <= 0 || P <= 0)
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
  if (NB > E-S) {
    NB = E-S;
  }

  Cd.step = C->step;
  Ad.step = A->step;
  Bd.step = A->step;

  if (flags & MTX_TRANSA) {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      Cd.md = &C->md[i*C->step+i];
      Ad.md = &A->md[i*A->step];
      // 1. update on diagonal
      _dmmat_rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
      Ad.md = &A->md[i*A->step];
      Bd.md = flags & MTX_LOWER ? &A->md[S*A->step] : &A->md[(i+nI)*A->step];
      nC = flags & MTX_LOWER ? i : E-i-nI;

      //_dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, vlen, &Acpy, &Bcpy);
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, KB, NB, MB, &Acpy, &Bcpy);
    }
  } else {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      Cd.md = &C->md[i*C->step+i];
      Ad.md = &A->md[i];
      // 1. update on diagonal
      _dmmat_rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
      Ad.md = &A->md[i];
      Bd.md = flags & MTX_LOWER ? &A->md[S] : &A->md[i+nI];
      nC = flags & MTX_LOWER ? i : E-i-nI;

      //_dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, &Acpy, &Bcpy);
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, KB, NB, MB, &Acpy, &Bcpy);
    }
  }
}


// SYR2K;
//   C = alpha*A*B.T + alpha*B*A.T + beta*C  
//   C = alpha*A.T*B + alpha*B.T*A + beta*C  if flags & MTX_TRANS
void dmmat_rank2_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                     double alpha, double beta, int flags,
                     int P, int S, int E,  int KB, int NB, int MB)
{
  register int i, j, nI, nC;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Acpy.md = Abuf;
  Acpy.step = MAX_KB;
  Bcpy.md = Bbuf;
  Bcpy.step = MAX_KB;

  if (E-S <= 0 || P <= 0)
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
  if (NB > E-S) {
    NB = E-S;
  }

  Cd.step = C->step;
  Ad.step = A->step;
  Bd.step = A->step;

  if (flags & MTX_TRANSA) {
    //   C = alpha*A.T*B + alpha*B.T*A + beta*C 
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      Cd.md = &C->md[i*C->step+i];
      Ad.md = &A->md[i*A->step];
      Bd.md = &B->md[i*B->step];
      // 1. update on diagonal
      _dmmat_rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
      Ad.md = &A->md[i*A->step];
      Bd.md = flags & MTX_LOWER ? &B->md[S*B->step] : &B->md[(i+nI)*B->step];
      nC = flags & MTX_LOWER ? i : E-i-nI;

      //_dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, vlen, &Acpy, &Bcpy);
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2nd part
      Cd.md = &C->md[i*C->step+i];
      Ad.md = &B->md[i*B->step];
      Bd.md = &A->md[i*A->step];
      // 1. update on diagonal
      _dmmat_rank_diag(&Cd, &Ad, &Bd, alpha, 1.0, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
      Ad.md = &B->md[i*B->step];
      Bd.md = flags & MTX_LOWER ? &A->md[S*A->step] : &A->md[(i+nI)*A->step];
      nC = flags & MTX_LOWER ? i : E-i-nI;

      //_dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, vlen, &Acpy, &Bcpy);
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, MTX_TRANSA, P, nC, nI, KB, NB, MB, &Acpy, &Bcpy);

    }
  } else {
    //   C = alpha*A*B.T + alpha*B*A.T + beta*C  
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      Cd.md = &C->md[i*C->step+i];
      Ad.md = &A->md[i];
      Bd.md = &B->md[i];
      // 1. update on diagonal
      _dmmat_rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
      Ad.md = &A->md[i];
      Bd.md = flags & MTX_LOWER ? &B->md[S] : &B->md[i+nI];
      nC = flags & MTX_LOWER ? i : E-i-nI;

      //_dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, &Acpy, &Bcpy);
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, KB, NB, MB, &Acpy, &Bcpy);

      Cd.md = &C->md[i*C->step+i];
      Ad.md = &B->md[i];
      Bd.md = &A->md[i];
      // 1. update on diagonal
      _dmmat_rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      Cd.md = flags & MTX_LOWER ? &C->md[i] : &C->md[(i+nI)*C->step+i];
      Ad.md = &B->md[i];
      Bd.md = flags & MTX_LOWER ? &A->md[S] : &A->md[i+nI];
      nC = flags & MTX_LOWER ? i : E-i-nI;

      //_dblock_mult_panel(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, &Acpy, &Bcpy);
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, MTX_TRANSB, P, nC, nI, KB, NB, MB, &Acpy, &Bcpy);
    }
  }
}



/*
 * update diagonal block
 *
 *  l00           a00 a01   b00 b01 b02    u00 u01 u02
 *  l10 l11       a10 a11   b10 b11 b12        u11 u12
 *  l20 l21 l22   a20 a21                          u22
 *
 */
 static void
_dmmat_trmupd_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                   double alpha, double beta,
                   int flags,  int P, int nC,
                   int KB, int NB, int MB,
                   mdata_t *Acpy, mdata_t *Bcpy)
                   //int vlen, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  register int i, incA, incB;
  mdata_t A0 = {A->md, A->step};
  mdata_t B0 = {B->md, B->step};

  incA = flags & MTX_TRANSA ? A->step : 1;
  incB = flags & MTX_TRANSB ? 1 : B->step;

  if (flags & MTX_UPPER) {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      dscale_vec(C->md, C->step, beta, nC-i);
      // update one row of C  (nC-i columns, 1 row)
      //_dblock_mult_panel(C, &A0, &B0, alpha, flags, P, nC-i, 1, vlen, Acpy, Bcpy);
      __dmult_inner_a_no_scale(C, &A0, &B0, alpha, flags, P, nC-i, 1, KB, NB, MB, Acpy, Bcpy);
      // move along the diagonal to next row of C
      C->md += C->step + 1;
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  } else {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      dscale_vec(C->md, C->step, beta, i+1);
      // update one row of C  (nC-i columns, 1 row)
      //_dblock_mult_panel(C, &A0, &B0, alpha, flags, P, i+1, 1, vlen, Acpy, Bcpy);
      __dmult_inner_a_no_scale(C, &A0, &B0, alpha, flags, P, i+1, 1, KB, NB, MB, Acpy, Bcpy);
      // move to next row of C
      C->md ++;
      // move A to next row
      A0.md += incA;
    }
  }
}

/*
 * Generic triangular matrix update:
 *      C = beta*C + alpha*A*B
 *      C = beta*C + alpha*A*B.T
 *      C = beta*C + alpha*A.T*B
 *      C = beta*C + alpha*A.T*B.T
 */
void dmmat_trmupd_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      double alpha, double beta, int flags,
                      int P, int S, int E, int KB, int NB, int MB)
{
  register int i, j, nI, nC, nR;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Acpy.md = Abuf;
  Acpy.step = MAX_KB;
  Bcpy.md = Bbuf;
  Bcpy.step = MAX_KB;

  if (E-S <= 0 || P <= 0)
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
  if (NB > E-S) {
    NB = E-S;
  }

  Cd.step = C->step;
  Ad.step = A->step;
  Bd.step = B->step;

  if (flags & MTX_UPPER) {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update block above the diagonal block
      Cd.md = &C->md[i*C->step];
      Ad.md = &A->md[S];
      Bd.md = flags & MTX_TRANSB ? &B->md[i] : &B->md[i*B->step];
      nR = i;
      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, flags, P, nI, nR, vlen, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, flags, P, nI, nR, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block on diagonal
      Cd.md = &C->md[i*C->step+i];
      Ad.md = flags & MTX_TRANSA ? &A->md[i*A->step] : &A->md[i];
      Bd.md = flags & MTX_TRANSB ? &B->md[i] : &B->md[i*B->step];
      _dmmat_trmupd_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

    }
  } else {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      Cd.md = &C->md[i*C->step+i];
      Ad.md = flags & MTX_TRANSA ? &A->md[i*A->step] : &A->md[i];
      Bd.md = flags & MTX_TRANSB ? &B->md[i] : &B->md[i*B->step];
      // 1. update on diagonal
      _dmmat_trmupd_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, KB, NB, MB, &Acpy, &Bcpy);

      // 2. update block below the diagonal block
      Cd.md = &C->md[i*C->step+i+nI];
      Ad.md = flags & MTX_TRANSA ? &A->md[(i+nI)*A->step] : &A->md[i+nI];
      Bd.md = flags & MTX_TRANSB ? &B->md[i] : &B->md[i*B->step];
      nR = E-i-nI;

      //_dmult_mm_intern(&Cd, &Ad, &Bd, alpha, flags, P, nI, nR, vlen, NB, NB, &Acpy, &Bcpy);
      __dmult_inner_a_no_scale(&Cd, &Ad, &Bd, alpha, flags, P, nI, nR, KB, NB, MB, &Acpy, &Bcpy);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
