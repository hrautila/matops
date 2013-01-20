
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "cmops.h"
#include "inner_dot.h"

// this will compute sub-block matrix product: Cij += Aik * Bkj using
// successive inner vector product (DOT) function.
void vpur_ddot(double *Cc, const double *Aroot, const double *Bc, double alpha,
               int ldC, int ldA, int ldB, int nSL, int nRE, int nVP)
{
  register int i, j;
  register double *c0, *c1, *c2, *c3;
  register const double *Br0, *Br1, *Br2, *Br3;
  const double *Ac;
  double *Cx = Cc;
  const double *Bx = Bc;

  //printf("ldA=%d, ldB=%d, ldC=%d, nSL=%d, nRE=%d, nVP=%d\n", ldA, ldB, ldC, nSL, nRE, nVP);
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

    for (i = 0; i < nRE; i++) {
      _inner_ddot4_sse(c0, c1, c2, c3, Ac, Br0, Br1, Br2, Br3, alpha, nVP);
      Ac += ldA;
      c0++;
      c1++;
      c2++;
      c3++;
    }
    // forward to start of next column in C, B
    Cc += 4*ldC;
    Bc += 4*ldB;
  }
  // Here if j == nSL --> nSL mod 4 == 0 and we are done
  // If work is divided right this should happen most of the time.
  if (j == nSL) {
    return;
  }

  // do the not-multiples of 4 cases....
  if (j < nSL-1) {
    Ac = Aroot;
    Br0 = Bc;
    Br1 = Br0 + ldB;
    c0 = Cc;
    c1 = c0 + ldC;
    for (i = 0; i < nRE; i++) {
      _inner_ddot2_sse(c0, c1, Ac, Br0, Br1, alpha, nVP);
      Ac += ldA;
      c0++;
      c1++;
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
    for (i = 0; i < nRE; i++) {
      _inner_ddot_sse(c0, Ac, Br0, alpha, nVP);
      Ac += ldA;
      c0++;
    }
    Cc += ldC;
    Bc += ldB;
  }
    
}

// nP is panel length
void dvpur_unaligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
                            double alpha, double beta,
                            int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB, nA;
  const double *Bc, *Ac, *AvpS;
  const double *Br0, *Br1, *Br2, *Br3;
  double *Cc, *c0, *c1, *c2, *c3;
  double Cpy[MAX_UA_NB*MAX_UA_MB]  __attribute__((aligned(16)));
  double Acpy[MAX_UA_VP*MAX_UA_MB] __attribute__((aligned(16)));
  double Bcpy[MAX_UA_VP*MAX_UA_NB] __attribute__((aligned(16)));


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

  while (vpS < nP) {
    nA = vpL - vpS;
    nA += (nA & 0x1);
    nB = nA;

    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[R*A->step + vpS];

    // copy (E-R) rows of (vpL-vpS) length
    colcpy(Acpy, nA, AvpS, A->step, vpL-vpS, E-R);
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
void dmult_unaligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                 double alpha, double beta,
                                 int P, int S, int L, int R, int E,
                                 int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_UA_NB || NB <= 0) {
    NB = MAX_UA_NB;
  }
  if (MB > MAX_UA_MB || MB <= 0) {
    MB = MAX_UA_MB;
  }
  if (vlen> MAX_UA_VP || vlen <= 0) {
    vlen = MAX_UA_VP;
  }

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dvpur_unaligned_transa(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}


void dvpur_aligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
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
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[R*A->step + vpS];

    vpur_ddot(Cc, AvpS, Bc, alpha, C->step, A->step, B->step, L-S, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > nP) {
      vpL = nP;
    }
  }
}


void dmult_aligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
                          double alpha, double beta,
                          int P, int S, int L, int R, int E,
                          int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      dvpur_aligned_transa(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:


