
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "cmops.h"
//#include "inner_axpy.h"

// this will compute sub-block matrix product: Cij += Aik * Bkj using
// successive vector scaling (AXPY) operations. Multipliers on a row.
void vpur_daxpy_trans(double *Cc, const double *Aroot, const double *Bc, double alpha,
                      int ldC, int ldA, int ldB, int nSL, int nRE, int nVP)
{
  register int j, k;
  register double *c0, *c1, *c2, *c3;
  register const double *Br0, *Br1, *Br2, *Br3;
  const double *Ac, *Bx, *Ax;
  
  Bx = Bc; Ax = Aroot;

  for (j = 0; j < nSL-3; j += 4) {
    Ac = Aroot;
    Br0 = Bc;
    Br1 = Br0 + 1;
    Br2 = Br1 + 1;
    Br3 = Br2 + 1;
    c0 = Cc;
    c1 = c0 + ldC;
    c2 = c1 + ldC;
    c3 = c2 + ldC;

    for (k = 0; k < nVP; k++) {
      // not sure if this really makes a difference
      __builtin_prefetch(Br0+ldB, 0, 1);
      __builtin_prefetch(Br1+ldB, 0, 1);
      __builtin_prefetch(Br2+ldB, 0, 1);
      __builtin_prefetch(Br3+ldB, 0, 1);

      _inner_daxpy4_sse(c0, c1, c2, c3, Ac, Br0, Br1, Br2, Br3, alpha, nRE);
      Br0 += ldB;
      Br1 += ldB;
      Br2 += ldB;
      Br3 += ldB;
      Ac += ldA;
    }
    // forward to start of next column in C, B
    Cc += 4*ldC;
    Bc += 4;
  }
  // Here if j == nSL --> nSL mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nSL)
    return;

  // do the not-multiples of 4 cases....
  if (j < nSL-1) {
    Ac = Aroot;
    Br0 = Bc;
    Br1 = Br0 + 1;
    c0 = Cc;
    c1 = c0 + ldC;
    for (k = 0; k < nVP; k++) {
      __builtin_prefetch(Br0+ldB, 0, 1);
      __builtin_prefetch(Br1+ldB, 0, 1);
      _inner_daxpy2_sse(c0, c1, Ac, Br0, Br1, alpha, nRE);
      Br0 += ldB;
      Br1 += ldB;
      Ac += ldA;
    }
    // forward to start of next column in C, B
    Cc += 2*ldC;
    Bc += 2;
    j += 2;
  }

  if (j < nSL) {
    // not multiple of 2
    Ac = Aroot;
    Br0 = Bc;
    c0 = Cc;
    for (k = 0; k < nVP; k++) {
      __builtin_prefetch(Br0+ldB, 0, 1);
      _inner_daxpy(c0, Ac, Br0,  alpha, nRE);
      Br0 += ldB;
      Ac += ldA;
    }
    Cc += ldC;
    Bc += 1;
    j++;
  }
    
}

// print copied tile
void print_tile(double *D, int ldD, int nR, int nC)
{
  register int i, j;
  for (i = 0; i < nR; i++) {
    printf("[");
    for (j = 0; j < nC; j++) {
      if (j != 0)
        printf(", ");
      printf("%9.2e", D[j*ldD+i]);
    }
    printf("]\n");
  }
}
// nP is panel length
void dvpur_unaligned_transb(mdata_t *C, const mdata_t *A, const mdata_t *B,
                            double alpha, double beta,
                            int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB, nA;
  const double *Bc, *Ac, *AvpS;
  const double *Br0, *Br1, *Br2, *Br3;
  double *Cc, *c0, *c1, *c2, *c3;
  double Cpy[MAX_NB_DDOT*MAX_MB_DDOT]  __attribute__((aligned(16)));
  double Acpy[MAX_VP_DDOT*MAX_MB_DDOT] __attribute__((aligned(16)));
  double Bcpy[MAX_VP_DDOT*MAX_NB_DDOT] __attribute__((aligned(16)));


  if (vlen > nP || vlen <= 0) {
    vlen = nP;
  }
  vpS = 0;
  vpL = vlen < nP ? vlen : nP;
  // row stride in Cpy 
  nC = E - R;
  nC += (nC & 0x1); // increments by 1 if not even.

  // Copy C block to local buffer
  Cc = &C->md[S*C->step+R];
  colcpy(Cpy, nC, Cc, C->step, E-R, L-S);

  // TODO: scaling with beta ....
  dscale_tile(Cpy, nC, beta, E-R, L-S);

  //nA = E - R;
  //nA += (nA & 0x1);
  while (vpS < nP) {
    nB = vpL - vpS;
    nB += (nB & 0x1);
    nA = nB;

    // column viewport start in panel B[:,S]
    Bc = &B->md[vpS*B->step + S];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    // copy (vpL-vpS) columns of (E-R) length

    // Copy A and B blocs and transpose on copy 
    colcpy_trans(Acpy, nA, AvpS, A->step, E-R, vpL-vpS);
    colcpy_trans(Bcpy, nB, Bc, B->step, L-S, vpL-vpS);

    vpur_ddot(Cpy, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > nP) {
      vpL = nP;
    }
  }
  // copy back.
  colcpy(Cc, C->step, Cpy, nC, E-R, L-S);
}

// Use this when rows of C and A are not aligned to 16bytes, ie C or A row strides
// are odd.
void dmult_unaligned_transb(mdata_t *C, const mdata_t *A, const mdata_t *B,
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
  if (vlen > MAX_VP_DDOT || vlen <= 0) {
    vlen = MAX_VP_DDOT;
  }

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dvpur_unaligned_transb(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}


void dvpur_aligned_transb(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          double alpha, double beta,
                          int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB, nA;
  const double *Bc, *Ac, *AvpS;
  const double *Br0, *Br1, *Br2, *Br3;
  double *Cc, *c0, *c1, *c2, *c3;

  if (vlen > nP || vlen <= 0) {
    vlen = nP;
  }
  vpS = 0;
  vpL = vlen < nP ? vlen : nP;
  // row stride in Cpy 
  nC = E - R;
  nC += (nC & 0x1); // increments by 1 if not even.

  Cc = &C->md[S*C->step+R];
  // TODO: scaling with beta ....
  dscale_tile(Cc, C->step, beta, E-R, L-S);

  while (vpS < nP) {
    nA = vpL - vpS;
    nA += (nA & 0x1);

    // column viewport start in panel B[:,S]
    Bc = &B->md[vpS*B->step + S];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    vpur_daxpy_trans(Cc, AvpS, Bc, alpha, C->step, A->step, B->step, L-S, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > nP) {
      vpL = nP;
    }
  }
}


void dmult_aligned_transb(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          double alpha, double beta,
                          int P, int S, int L, int R, int E,
                          int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dvpur_aligned_transb(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:


