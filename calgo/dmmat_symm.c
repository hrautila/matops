
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "cmops.h"
#include "colcpy.h"

extern
void _dmmat_mult_diag(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      double alpha, int flags, 
                      int nP, int nSL, int nRE, int vlen, mdata_t *Acpy, mdata_t *Bcpy);

extern
void __dmult_inner_a_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              double alpha, int flags,
                              int P, int nSL, int nR, 
                              int KB, int NB, int MB, mdata_t *Acpy, mdata_t *Bcpy);


// C += A*B; A is the diagonal block
void _dmmat_mult_diag(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       double alpha, int flags, 
                       int nP, int nSL, int nRE, int vlen, mdata_t *Acpy, mdata_t *Bcpy)
{
  int unit = flags & MTX_UNIT ? 1 : 0;
  int nA, nB, nAC;

  if (nP == 0)
    return;

  //nA = nRE + (nRE & 0x1);
  //nB = nA;
  nAC = flags & MTX_RIGHT ? nSL : nRE;
  
  //printf("nAC=%d, nSL=%d, nP=%d, nRE=%d\n", nAC, nSL, nP, nRE);

  //printf("1. pre-rank: A=\n"); print_tile(A->md, A->step, nAC, nAC);
  if (flags & MTX_LOWER) {
    // upper part of source untouchable, copy diagonal block and fill upper part
    colcpy_fill_up(Acpy->md, Acpy->step, A->md, A->step, nAC, nAC, unit);
  } else {
    // lower part of source untouchable, copy diagonal block and fill lower part
    colcpy_fill_low(Acpy->md, Acpy->step, A->md, A->step, nAC, nAC, unit);
  }
  //printf("1b. Acpy=\n"); print_tile(Acpy->md, Acpy->step, nAC, nAC);

  if (flags & MTX_RIGHT) {
    colcpy_trans(Bcpy->md, Bcpy->step, B->md, B->step, nRE, nSL);
  } else {
    colcpy(Bcpy->md, Bcpy->step, B->md, B->step, nRE, nSL);
  }
  //printf("1c. Bcpy=\n"); print_tile(Bcpy->md, Bcpy->step, nRE, nSL);

  if (flags & MTX_RIGHT) {
    __dmult_blk_inner_a(C, Bcpy, Acpy, alpha, nAC, nRE, nP);
  } else {
    __dmult_blk_inner_a(C, Acpy, Bcpy, alpha, nSL, nAC, nP);
  }
  //printf("2b. post update: C=\n"); print_tile(C->md, C->step, nRE, nSL);

}



void dmult_symm_blocked3(mdata_t *C, const mdata_t *A, const mdata_t *B,
                         double alpha, double beta, int flags,
                         int P, int S, int L, int R, int E,
                         int KB, int NB, int MB)
{
  int i, j, nI, nJ, flags1, flags2;
  mdata_t A0, B0, C0, Acpy, Bcpy, *Ap, *Bp;

  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Acpy.md = Abuf;
  Acpy.step = MAX_KB;
  Bcpy.md = Bbuf;
  Bcpy.step = MAX_KB;

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

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

  C0.step = C->step;
  A0.step = A->step;
  B0.step = B->step;

  flags1 = 0;
  flags2 = 0;

  // C = A*B; Multiply with A from LEFT is default.
  if (flags & MTX_LEFT || !(flags & MTX_RIGHT)) {
    /*
      P is A, B common dimension, e.g. P cols in A and P rows in B.
    
      [R,R] [E,E] define block on A diagonal that divides A in three blocks
      if A is upper:
        A0 [0, R] [R, E]; B0 [0, S] [R, L] (R rows,cols in P); (A transposed)
        A1 [R, R] [E, E]; B1 [R, S] [E, L] (E-R rows,cols in P)
        A2 [R, E] [E, N]; B2 [E, S] [N, L] (N-E rows, cols in  P)
      if A is LOWER:
        A0 [R, 0] [E, R]; B0 [0, S] [R, L]
        A1 [R, R] [E, E]; B1 [R, S] [E, L] (diagonal block, fill_up);
        A2 [E, R] [E, N]; B2 [E, S] [N, L] (A transpose)
        
      C = A0*B0 + A1*B1 + A2*B2
    */
    flags1 |= flags & MTX_UPPER ? MTX_TRANSA : 0;
    flags2 |= flags & MTX_LOWER ? MTX_TRANSA : 0;

    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;

      // for all column of C, B ...
      for (j = S; j < L; j += NB) {
        nJ = L - j < NB ? L - j : NB;
        C0.md = &C->md[j*C->step + i];
        //printf("** i=%d, j=%d, nI=%d, nJ=%d, P=%d\n", i, j, nI, nJ, P);
      
        // block of C upper left at [i,j], lower right at [i+nI, j+nj]
        dscale_tile(C0.md, C0.step, beta, nI, nJ);

        // 1. off diagonal block in A; if UPPER then above [i,j]; if LOWER then left of [i,j]
        // above|left diagonal
        A0.md = flags & MTX_UPPER ? &A->md[i*A->step] : &A->md[i];
        B0.md = &B->md[j*B->step];
        __dmult_inner_a_no_scale(&C0, &A0, &B0, alpha, flags1, i, nJ, nI, KB, NB, MB, &Acpy, &Bcpy);
        //printf("--- pre diag: C=\n"); print_tile(C->md, C->step, E-R, L-S);

        // 2. on-diagonal block in A;
        // diagonal block
        A0.md = &A->md[i*A->step + i];
        B0.md = &B->md[j*B->step + i];
        _dmmat_mult_diag(&C0, &A0, &B0, alpha, flags, nI, nJ, nI, KB, &Acpy, &Bcpy);
        //printf("+++ post diag: C=\n"); print_tile(C->md, C->step, E-R, L-S);

        // 3. off-diagonal block in A; if UPPER then right of [i, i+nI]; if LOWER then below [i+nI, i]
        // right|below of diagonal
        A0.md = flags & MTX_UPPER ? &A->md[(i+nI)*A->step + i] : &A->md[i*A->step + i+nI];
        B0.md = &B->md[j*B->step + i+nI];
        __dmult_inner_a_no_scale(&C0, &A0, &B0, alpha, flags2, P-i-nI, nJ, nI, KB, NB, MB, &Acpy, &Bcpy);
        //printf("=== EOL: C=\n"); print_tile(C->md, C->step, E-R, L-S);
      }
    }
  } else {

    /*
      P is A, B common dimension, e.g. P cols in A and P rows in B.
    
      C = B * A;
      [S,S] [L,L] define block on A diagonal that divides A in three blocks
      if A is upper:
        A0 [0, S] [S, S]; B0 [R, 0] [E, S] (R rows,cols in P); (A transposed)
        A1 [S, S] [L, L]; B1 [R, S] [E, L] (E-R rows,cols in P)
        A2 [S, L] [L, N]; B2 [R, L] [E, N] (N-E rows, cols in  P)
      if A is LOWER:
        A0 [S, 0] [S, S]; B0 [R, 0] [E, S]
        A1 [S, S] [L, L]; B1 [R, S] [E, L] (diagonal block, fill_up);
        A2 [L, S] [N, L]; B2 [R, L] [E, N] (A transpose)
        
      C = A0*B0 + A1*B1 + A2*B2
    */

    register int nR, nC, ic, ir;
    flags1 = flags & MTX_TRANSB ? MTX_TRANSA : 0;
    flags2 = flags & MTX_TRANSB ? MTX_TRANSA : 0;

    flags1 |= flags & MTX_LOWER ? MTX_TRANSB : 0;
    flags2 |= flags & MTX_UPPER ? MTX_TRANSB : 0;

    for (ic = S; ic < L; ic += NB) {
      nC = L - ic < NB ? L - ic : NB;

      // for all rows of C, B ...
      for (ir = R; ir < E; ir += MB) {
        nR = E - ir < MB ? E - ir : MB;

        C0.md = &C->md[ic*C->step + ir];
        //printf("** ir=%d, ic=%d, nR=%d, nC=%d, P=%d\n", ir, ic, nR, nC, P);
      
        dscale_tile(C0.md, C0.step, beta, nR, nC);

        // above|left diagonal
        A0.md = flags & MTX_UPPER ? &A->md[ic*A->step] : &A->md[ic];
        B0.md = &B->md[ir];
        __dmult_inner_a_no_scale(&C0, &B0, &A0, alpha, flags1, ic, nC, nR, KB, NB, MB, &Acpy, &Bcpy);
        //printf("--- pre diag: C=\n"); print_tile(C->md, C->step, E-R, L-S);

        // diagonal block
        A0.md = &A->md[ic*A->step + ic];
        B0.md = &B->md[ic*B->step+ir];
        _dmmat_mult_diag(&C0, &A0, &B0, alpha, flags, nC, nC, nR, KB, &Acpy, &Bcpy);
        //printf("+++ post diag: C=\n"); print_tile(C->md, C->step, E-R, L-S);

        // right|below of diagonal
        A0.md = flags & MTX_UPPER ? &A->md[(ic+nC)*A->step + ic] : &A->md[ic*A->step +ic+nC];
        B0.md = &B->md[(ic+nC)*B->step+ir];
        __dmult_inner_a_no_scale(&C0, &B0, &A0, alpha, flags2, P-ic-nC, nC, nR, KB, NB, MB, &Acpy, &Bcpy);
        //printf("=== EOL: C=\n"); print_tile(C->md, C->step, E-R, L-S);
      }
    }
  }
}




// Local Variables:
// indent-tabs-mode: nil
// End:
