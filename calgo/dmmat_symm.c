
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "cmops.h"
#include "colcpy.h"

extern void
_dblock_ddot_sse(double *Cc, const double *Aroot, const double *Bc, double alpha,
                 int ldC, int ldA, int ldB, int nSL, int nRE, int nVP);


void _dblock_symm_cpy(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      double alpha, double beta, int flags,
                      int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB, nA;
  const double *Bc, *Ac, *AvpS;
  //const double *Br0, *Br1, *Br2, *Br3;
  double *Cc; //, *c0, *c1, *c2, *c3;
  //double Cpy[MAX_NB_DDOT*MAX_MB_DDOT]  __attribute__((aligned(16)));
  double Acpy[MAX_VP_DDOT*MAX_MB_DDOT] __attribute__((aligned(16)));
  double Bcpy[MAX_VP_DDOT*MAX_NB_DDOT] __attribute__((aligned(16)));
  int unit = flags & MTX_UNIT ? 1 : 0;


  if (vlen > nP || vlen <= 0) {
    vlen = nP;
  }

  //printf("0. nP=%d, L=%d, S=%d, E=%d, R=%d, vlen=%d\n", nP, L, S, E, R, vlen);

  // row stride in Cpy 
  //nC = E - R;
  //nC += (nC & 0x1); // increments by 1 if not even.

  // Copy C block to local buffer
  Cc = &C->md[S*C->step+R];
  //colcpy(Cpy, nC, Cc, C->step, E-R, L-S);
  nC = C->step;

  // TODO: scaling with beta ....
  dscale_tile(Cc, nC, beta, E-R, L-S);

  // nP is columns in A, 
  vpS = 0;
  vpL = vlen < R ? vlen : R;

  // 1. this is the panel left of diagonal block (LOWER) or above the diagonal (UPPER) 
  //    vps, vpL < R: work it like notrans
  while (vpS < R) {
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;
    //printf("1. vpS=%d, vpL=%d, nC=%d, nB=%d, nA=%d\n", vpS, vpL, nC, nB, nA);

    // viewport starts in B, A
    if (flags & MTX_LOWER) {
      // The panel left of diagonal block
      Bc = &B->md[S*B->step + vpS];
      AvpS = &A->md[vpS*A->step + R];
      // transpose A on copy to be able to DOT operations.
      colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
      colcpy4_trans(Acpy, nA, AvpS, A->step, E-R, vpL-vpS);
    } else {
      // The panel above of diagonal block
      Bc = &B->md[S*B->step + vpS];
      AvpS = &A->md[R*A->step + vpS];

      colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
      colcpy(Acpy, nA, AvpS, A->step, vpL-vpS, E-R); //, vpL-vpS);
    }
    //printf("1. update: A=\n"); print_tile(Acpy, nA, vpL-vpS, E-R);
    //printf("1. update: B=\n"); print_tile(Bcpy, nB, vpL-vpS, L-S);

    if (flags & MTX_LEFT) {
      _dblock_ddot_sse(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
    } else {
    }
    //printf("1. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

    vpS = vpL;
    vpL += vlen;
    if (vpL > R) {
      vpL = R;
    }
  }
  
  // 2. this is the diagonal block, with upper part untouchable
  //    R <= vps, vpL < E: diagonal part, copy_and_fill
  //    here vpS == R, update vpL += E - R as this block is square, diagonal
  vpL += E-R;
  while (vpS < E) {
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;

    //printf("2. vpS=%d, vpL=%d, nC=%d, nB=%d, nA=%d\n", vpS, vpL, nC, nB, nA);
    // viewport starts in B, A
    if (flags & MTX_LOWER) {
      // upper part of source untouchable, copy diagonal block and fill upper part
      Bc = &B->md[S*B->step + vpS];
      AvpS = &A->md[vpS*A->step + R];
      //print_tile(Acpy, nA, E-R, E-R);
      colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
      colcpy_fill_up(Acpy, nA, AvpS, A->step, E-R, E-R, unit);
    } else {
      // lower part of source untouchable, copy diagonal block and fill lower part
      Bc = &B->md[S*B->step + vpS];
      AvpS = &A->md[vpS*A->step + R];
      colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
      colcpy_fill_low(Acpy, nA, AvpS, A->step, E-R, E-R, unit);
    }

    if (flags & MTX_LEFT) {
      _dblock_ddot_sse(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
    }
    //printf("2. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

    vpS = vpL;
    vpL += vlen;
    if (vpL > E) {
      vpL = E;
    }
  }

  // 3. this is rest of the panel rows below or right of diagonal block.
  //    vps, vpL >= E && < nP: rest of the row like transA case.
  //    here vpS == E, and vpL == vpS + vlen
  vpL = vpS + vlen;
  if (vpL > nP) {
    vpL = nP;
  }
  while (vpS < nP) {
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;

    //printf("3. vpS=%d, vpL=%d, nC=%d, nB=%d, nA=%d\n", vpS, vpL, nC, nB, nA);
    if (flags & MTX_LOWER) {
      // this is rest of the panel rows below of the diagonal block.
      Bc = &B->md[S*B->step + vpS];
      AvpS = &A->md[R*A->step + vpS];
      colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
      colcpy(Acpy, nA, AvpS, A->step, vpL-vpS, E-R);
    } else {
      // this is rest of the panel rows right of the diagonal block.
      Bc = &B->md[S*B->step + vpS];
      AvpS = &A->md[vpS*A->step + R];

      // transpose A on copy to be able to DOT operations.
      colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
      colcpy4_trans(Acpy, nA, AvpS, A->step, E-R, vpL-vpS);
    }

    if (flags & MTX_LEFT) {
      // C += alpha * A * B
      _dblock_ddot_sse(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
    } else {
      // C += alpha * B * A
    }
    //printf("3. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

    vpS = vpL;
    vpL += vlen;
    if (vpL > nP) {
      vpL = nP;
    }
  }
  // copy back.
  //colcpy(Cc, C->step, Cpy, nC, E-R, L-S);
}


void dmult_symm_blocked(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        double alpha, double beta, int flags,
                        int P, int S, int L, int R, int E,
                        int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB_DDOT || NB <= 0) {
    NB = MAX_NB_DDOT;
  }
  if (MB > MAX_MB_DDOT || MB <= 0) {
    MB = MAX_MB_DDOT;
  }
  if (vlen> MAX_VP_DDOT || vlen <= 0) {
    vlen = MAX_VP_DDOT;
  }

  // A is square matrix: 
  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      _dblock_symm_cpy(C, A, B, alpha, beta, flags, P, j, j+nJ, i, i+nI, vlen);
      //printf("\nC=\n"); print_tile(C->md, C->step, E-R, L-S);
    }
  }
}



// C += A*B; A is the diagonal block
void _dblock_mult_diag(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       double alpha, int flags, 
                       int nP, int nSL, int nRE, int vlen, cbuf_t *Acpy, cbuf_t *Bcpy)
{
  // assert (nSL == nRE)
  int unit = flags & MTX_UNIT ? 1 : 0;
  int nA, nB;

  if (nP == 0)
    return;

  nA = nRE + (nRE & 0x1);
  nB = nA;

  if (flags & MTX_LOWER) {
    // upper part of source untouchable, copy diagonal block and fill upper part
    colcpy_fill_up(Acpy->data, nA, A->md, A->step, nRE, nRE, unit);
  } else {
    // lower part of source untouchable, copy diagonal block and fill lower part
    colcpy_fill_low(Acpy->data, nA, A->md, A->step, nRE, nRE, unit);
  }

  if (flags & MTX_RIGHT) {
    colcpy_trans(Bcpy->data, nB, B->md, B->step, nRE, nSL);
  } else {
    colcpy(Bcpy->data, nB, B->md, B->step, nRE, nSL);
  }
  if (flags & MTX_RIGHT) {
    _dblock_ddot_sse(C->md, Bcpy->data, Acpy->data, alpha, C->step, nB, nA, nSL, nRE, nP);
  } else {
    _dblock_ddot_sse(C->md, Acpy->data, Bcpy->data, alpha, C->step, nA, nB, nSL, nRE, nP);
  }
  //printf("2. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

}

void dmult_symm_blocked2(mdata_t *C, const mdata_t *A, const mdata_t *B,
                         double alpha, double beta, int flags,
                         int P, int S, int L, int R, int E,
                         int vlen, int NB, int MB)
{
  int i, j, nI, nJ, aflags;
  mdata_t A0, B0, C0;
  double Abuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  double Bbuf[MAX_VP_ROWS*MAX_VP_COLS] __attribute__((aligned(16)));
  cbuf_t Acpy = {Abuf, MAX_VP_ROWS*MAX_VP_COLS};
  cbuf_t Bcpy = {Bbuf, MAX_VP_ROWS*MAX_VP_COLS};

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_VP_COLS || NB <= 0) {
    NB = MAX_VP_COLS;
  }
  if (MB > MAX_VP_ROWS || MB <= 0) {
    MB = MAX_VP_ROWS;
  }
  if (vlen > MAX_VP_ROWS || vlen <= 0) {
    vlen = MAX_VP_ROWS;
  }

  C0.step = C->step;
  A0.step = A->step;
  B0.step = B->step;
  // A is square matrix: 
  if (flags & MTX_RIGHT) {
    
  } else {
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

    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;

      // for all column of C, B ...
      for (j = S; j < L; j += NB) {
        nJ = L - j < NB ? L - j : NB;
        C0.md = &C->md[j*C->step + i];

        //printf("i: %d, j: %d, nI: %d, nJ: %d, A2.r: %d\n", i, j, nI, nJ, P-i-nI);
        dscale_tile(C0.md, C0.step, beta, nI, nJ);

        // above|left diagonal
        A0.md = flags & MTX_UPPER ? &A->md[i*A->step] : &A->md[i];
        B0.md = &B->md[j*B->step];
        //printf("..A0:\n"); print_tile(A0.md, A0.step, i, nI);
        //printf("..B0:\n"); print_tile(B0.md, B0.step, i,  nJ);
        aflags = flags & MTX_UPPER ? MTX_TRANSA : MTX_NOTRANS;
        _dblock_mult_panel(&C0, &A0, &B0, alpha, aflags, i, nJ, nI, vlen, &Acpy, &Bcpy);

        // diagonal block
        A0.md = &A->md[i*A->step + i];
        B0.md = &B->md[j*B->step + i];
        //printf("..A1:\n"); print_tile(A0.md, A0.step, nI, nI);
        //printf("..B1:\n"); print_tile(B0.md, B0.step, nI,  nJ);
        _dblock_mult_diag(&C0, &A0, &B0, alpha, flags, nI, nJ, nI, vlen, &Acpy, &Bcpy);

        // right|below of diagonal
        A0.md = flags & MTX_UPPER ? &A->md[(i+nI)*A->step + i] : &A->md[i*A->step + i+nI];
        B0.md = &B->md[j*B->step + i+nI];
        //printf("..A2:\n"); print_tile(A0.md, A0.step, nI, P-i-nI);
        //printf("..B2:\n"); print_tile(B0.md, B0.step, P-i-nI, nJ);
        aflags = flags & MTX_UPPER ? MTX_NOTRANS : MTX_TRANSA;
        _dblock_mult_panel(&C0, &A0, &B0, alpha, aflags, P-i-nI, nJ, nI, vlen, &Acpy, &Bcpy);
        //printf("..C:\n"); print_tile(C->md, C->step, E-R, L-S);
      }
    }
  }
}




// Local Variables:
// indent-tabs-mode: nil
// End:
