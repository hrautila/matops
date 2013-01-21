
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "cmops.h"
#include "inner_axpy.h"


// This will compute sub-block matrix product: Cij += Aik * Bkj using succesive
// vector scaling (AXPY) operations.
void vpur_daxpy(double *Cc, const double *Aroot, const double *Bc, double alpha,
                int ldC, int ldA, int ldB, int nSL, int nRE, int nVP)
{
  register int j, k;
  register double *c0, *c1, *c2, *c3;
  register const double *Br0, *Br1, *Br2, *Br3;
  const double *Ac;

  for (j = 0; j < nSL-3; j += 4) {
    Ac = Aroot;
    Br0 = Bc;
    Br1 = Br0 + ldB;
    Br2 = Br1 + ldB;
    Br3 = Br2 + ldB;
    c0 = Cc;
    c1 = c0 + ldC;
    c2 = c1 + ldC;
    c3 = c2 + ldC;

    for (k = 0; k < nVP; k++) {
      _inner_daxpy4_sse(c0, c1, c2, c3, Ac, Br0, Br1, Br2, Br3, alpha, nRE);
      Br0++;
      Br1++;
      Br2++;
      Br3++;
      Ac += ldA;
    }
    // forward to start of next column in C, B
    Cc += 4*ldC;
    Bc += 4*ldB;
  }
  // Here if j == nSL --> nSL mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nSL)
    return;

  // do the not-multiples of 4 cases....
  if (j < nSL-1) {
    Ac = Aroot;
    Br0 = Bc;
    Br1 = Br0 + ldB;
    c0 = Cc;
    c1 = c0 + ldC;
    for (k = 0; k < nVP; k++) {
      _inner_daxpy2_sse(c0, c1, Ac, Br0, Br1, alpha, nRE);
      Br0++;
      Br1++;
      Ac += ldA;
    }
    // forward to start of next column in C, B
    Cc += 2*ldC;
    Bc += 2*ldB;
    j += 2;
  }

  if (j < nSL) {
    // not multiple of 2
    Ac = Aroot;
    Br0 = Bc;
    c0 = Cc;
    for (k = 0; k < nVP; k++) {
      _inner_daxpy_sse(c0, Ac, Br0,  alpha, nRE);
      Br0++;
      Ac += ldA;
    }
    Cc += ldC;
    Bc += ldB;
  }
    
}

// unrolling columns and rows within viewport with data aligned at 16 bytes.
// This will compute matrix block defined by A[R:E,:] row panel and B[:,S:L] column panel
// as series of sub-block matrix multplications. If A and B panels are divided to sub-blocks A[k], B[k]
// k = 0 ... P/vlen then C is sum of A[k]*B[k].
void dvpur_aligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                           double alpha, double beta,
                           int P, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL;
  const double *Bc, *Ac, *AvpS;
  double *Cc;

  vpS = 0;
  vpL = vlen < P ? vlen : P;

  // block start C[R, S]
  Cc = &C->md[S*C->step+R];

  // scaling with beta ...
  dscale_tile(Cc, C->step, beta, E-R, L-S);

  while (vpS < P) {
    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    vpur_daxpy(Cc, AvpS, Bc, alpha, C->step, A->step, B->step, L-S, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > P) {
      vpL = P;
    }
  }
}


// Use this when rows of C and A are aligned to 16bytes, ie C and A row strides
// are even.
void dmult_aligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, double beta,
                           int P, int S, int L, int R, int E,
                           int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (vlen > MAX_VP || vlen <= 0) {
    vlen = MAX_VP;
  }

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dvpur_aligned_notrans(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// nP is panel length
void dvpur_unaligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, double beta,
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
    nB = vpL-vpS;
    nB += (nB & 0x1);
    nA = nB;

    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    // transpose A on copy to be able to DOT operations.
    colcpy_trans(Acpy, nA, AvpS, A->step, E-R, vpL-vpS);
    colcpy(Bcpy, nB, Bc, B->step, vpL-vpS, L-S);

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
void dmult_unaligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
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

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dvpur_unaligned_notrans(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:
