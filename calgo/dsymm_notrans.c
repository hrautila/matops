
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "cmops.h"
#include "colcpy.h"

void dvpur_symm_ua_up_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              double alpha, double beta,
                              int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB, nA;
  const double *Bc, *Ac, *AvpS;
  const double *Br0, *Br1, *Br2, *Br3;
  double *Cc, *c0, *c1, *c2, *c3;
  //double Cpy[MAX_NB_DDOT*MAX_MB_DDOT]  __attribute__((aligned(16)));
  double Acpy[MAX_VP_DDOT*MAX_MB_DDOT] __attribute__((aligned(16)));
  double Bcpy[MAX_VP_DDOT*MAX_NB_DDOT] __attribute__((aligned(16)));


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

  // 1. this is the panel above diagonal block, 
  //    vps, vpL < R: work it like transA
  while (vpS < R) {
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;
    //printf("1. vpS=%d, vpL=%d, nC=%d, nB=%d, nA=%d\n", vpS, vpL, nC, nB, nA);
    // viewport starts in B, A
    Bc = &B->md[S*B->step + vpS];
    AvpS = &A->md[R*A->step + vpS];

    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
    colcpy(Acpy, nA, AvpS, A->step, vpL-vpS, E-R); //, vpL-vpS);
    //printf("1. update: A=\n"); print_tile(Acpy, nA, vpL-vpS, E-R);
    //printf("1. update: B=\n"); print_tile(Bcpy, nB, vpL-vpS, L-S);

    vpur_ddot(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
    //printf("1. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

    vpS = vpL;
    vpL += vlen;
    if (vpL > R) {
      vpL = R;
    }
  }
  
  // 2. this is the diagonal block, with lower part untouchable
  //    R <= vps, vpL < E: diagonal part, copy_and_fill
  //    here vpS == R, update vpL += E - R as this block is square, diagonal
  vpL += E-R;
  while (vpS < E) {
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;

    //printf("2. vpS=%d, vpL=%d, nC=%d, nB=%d, nA=%d\n", vpS, vpL, nC, nB, nA);
    // viewport starts in B, A
    Bc = &B->md[S*B->step + vpS];
    AvpS = &A->md[vpS*A->step + R];

    // copy part of diagonal block and fill lower part
    //print_tile(Acpy, nA, E-R, E-R);
    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
    colcpy_fill_low(Acpy, nA, AvpS, A->step, E-R, E-R);

    vpur_ddot(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
    //printf("2. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

    vpS = vpL;
    vpL += vlen;
    if (vpL > E) {
      vpL = E;
    }
  }

  // 3. this is rest of the panel rows on diagonal block.
  //    vps, vpL >= E && < nP: rest of the row like notrans case.
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
    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    // transpose A on copy to be able to DOT operations.
    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
    colcpy4_trans(Acpy, nA, AvpS, A->step, E-R, vpL-vpS);

    vpur_ddot(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
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

// Use this when rows of C and A are not aligned to 16bytes, ie C or A row strides
// are odd.
void dmult_symm_ua_up_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              double alpha, double beta,
                              int P, int S, int L, int R, int E,
                              int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

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
      dvpur_symm_ua_up_notrans(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
      //printf("\nC=\n");
      //print_tile(C->md, C->step, E-R, L-S);
      //printf("\n");
    }
  }
}


void dvpur_symm_ua_low_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                               double alpha, double beta,
                               int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB, nA;
  const double *Bc, *Ac, *AvpS;
  const double *Br0, *Br1, *Br2, *Br3;
  double *Cc, *c0, *c1, *c2, *c3;
  //double Cpy[MAX_NB_DDOT*MAX_MB_DDOT]  __attribute__((aligned(16)));
  double Acpy[MAX_VP_DDOT*MAX_MB_DDOT] __attribute__((aligned(16)));
  double Bcpy[MAX_VP_DDOT*MAX_NB_DDOT] __attribute__((aligned(16)));


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

  // 1. this is the panel left of diagonal block, 
  //    vps, vpL < R: work it like notrans
  while (vpS < R) {
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;
    //printf("1. vpS=%d, vpL=%d, nC=%d, nB=%d, nA=%d\n", vpS, vpL, nC, nB, nA);
    // viewport starts in B, A
    Bc = &B->md[S*B->step + vpS];
    AvpS = &A->md[vpS*A->step + R];

    // transpose A on copy to be able to DOT operations.
    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
    colcpy4_trans(Acpy, nA, AvpS, A->step, E-R, vpL-vpS);
    //printf("1. update: A=\n"); print_tile(Acpy, nA, vpL-vpS, E-R);
    //printf("1. update: B=\n"); print_tile(Bcpy, nB, vpL-vpS, L-S);

    vpur_ddot(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
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
    Bc = &B->md[S*B->step + vpS];
    AvpS = &A->md[vpS*A->step + R];
    // copy part of diagonal block and fill upper part
    //print_tile(Acpy, nA, E-R, E-R);
    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
    colcpy_fill_up(Acpy, nA, AvpS, A->step, E-R, E-R);

    vpur_ddot(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
    //printf("2. post update: C=\n"); print_tile(Cpy, nC, E-R, L-S);

    vpS = vpL;
    vpL += vlen;
    if (vpL > E) {
      vpL = E;
    }
  }

  // 3. this is rest of the panel rows on below diagonal block.
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
    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[R*A->step + vpS];

    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);
    colcpy(Acpy, nA, AvpS, A->step, vpL-vpS, E-R); 

    vpur_ddot(Cc, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);
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


void dmult_symm_ua_low_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                               double alpha, double beta,
                               int P, int S, int L, int R, int E,
                               int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

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
      dvpur_symm_ua_low_notrans(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}




// Local Variables:
// indent-tabs-mode: nil
// End:
