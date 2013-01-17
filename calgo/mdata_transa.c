
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <x86intrin.h>

#include "cmops.h"

static void inner_loop4_sse(double *c0, double *c1, double *c2, double *c3,
                            const double *Ar, const double *b0, const double *b1, const double *b2,
                            const double *b3, double alpha, int nVP)
{
  register int k;
  register double f0, f1, cval;
  register __m128d AR, B0, B1, C0, C1, C2, C3, F0, F1, ALP;

  C0 = _mm_set1_pd(0.0);
  C1 = _mm_set1_pd(0.0);
  C2 = _mm_set1_pd(0.0);
  C3 = _mm_set1_pd(0.0);
  ALP = _mm_set1_pd(alpha);

  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    B0 = _mm_load_pd(b2);
    B1 = _mm_load_pd(b3);
    F0 = AR * B0;
    C2 = C2 + F0;
    F1 = AR * B1;
    C3 = C3 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
    b2 += 2;
    b3 += 2;

    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    B0 = _mm_load_pd(b2);
    B1 = _mm_load_pd(b3);
    F0 = AR * B0;
    C2 = C2 + F0;
    F1 = AR * B1;
    C3 = C3 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
    b2 += 2;
    b3 += 2;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    B0 = _mm_load_pd(b2);
    B1 = _mm_load_pd(b3);
    F0 = AR * B0;
    C2 = C2 + F0;
    F1 = AR * B1;
    C3 = C3 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
    b2 += 2;
    b3 += 2;
    k += 2;
  }
  if (k < nVP) {
    cval = Ar[0] * alpha;
    f0 = cval * b0[0];
    c0[0] += f0;
    f1 = cval * b1[0];
    c1[0] += f1;
    f0 = cval * b2[0];
    c2[0] += f0;
    f1 = cval * b2[0];
    c3[0] += f1;
    k++;
  }
 update:
  C0 = C0 * ALP;
  c0[0] += C0[0];
  c0[0] += C0[1];
  C1 = C1 * ALP;
  c1[0] += C1[0];
  c1[0] += C1[1];
  C2 = C2 * ALP;
  c2[0] += C2[0];
  c2[0] += C2[1];
  C3 = C3 * ALP;
  c3[0] += C3[0];
  c3[0] += C3[1];
}

static void inner_loop2_sse(double *c0, double *c1,
                            const double *Ar, const double *b0, const double *b1, 
                            double alpha, int nVP)
{
  register int k;
  register double f0, f1, cval;
  register __m128d AR, B0, B1, C0, C1, F0, F1, ALP;

  C0 = _mm_set1_pd(0.0);
  C1 = _mm_set1_pd(0.0);
  ALP = _mm_set1_pd(alpha);

  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;

    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    B0 = _mm_load_pd(b0);
    B1 = _mm_load_pd(b1);
    F0 = AR * B0;
    C0 = C0 + F0;
    F1 = AR * B1;
    C1 = C1 + F1;
    Ar += 2;
    b0 += 2;
    b1 += 2;
    k += 2;
  }
  if (k < nVP) {
    cval = Ar[0] * alpha;
    f0 = cval * b0[0];
    c0[0] += f0;
    f1 = cval * b1[0];
    c1[0] += f1;
    k++;
  }
 update:
  C0 = C0 * ALP;
  c0[0] += C0[0];
  c0[0] += C0[1];
  C1 = C1 * ALP;
  c1[0] += C1[0];
  c1[0] += C1[1];
}

static void inner_loop_sse(double *Cr, const double *Ar, const double *Br, double alpha, int nVP)
{
  register int k;
  register double f0, cval;
  register __m128d AR, BR, C0, F0, ALP;
  double zero = 0.0;

  C0 = _mm_set1_pd(0.0);
  ALP = _mm_set1_pd(alpha);

  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    AR = _mm_load_pd(Ar);
    BR = _mm_load_pd(Br);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2;

    AR = _mm_load_pd(Ar);
    BR = _mm_load_pd(Br);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    AR = _mm_load_pd(Ar);
    BR = _mm_load_pd(Br);
    F0 = AR * BR;
    C0 = C0 + F0;
    Ar += 2;
    Br += 2;
    k += 2;
  }
  if (k < nVP) {
    cval = Ar[0] * Br[0];
    Cr[0] += cval * alpha;
    Br++;
    Ar++;
    k++;
  }
 update:
  C0 = C0 * ALP;
  Cr[0] += C0[0];
  Cr[0] += C0[1];
}

static void inner_loop(double *Cr, const double *Ar, const double *Br, double alpha, int nVP)
{
  int k;
  double f0, f1, f2, f3, cval;

  cval = 0.0;
  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    f1 = Ar[1] * Br[1];
    cval += f1;
    f2 = Ar[2] * Br[2];
    cval += f2;
    f3 = Ar[3] * Br[3];
    cval += f3;
    Br += 4;
    Ar += 4;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    f1 = Ar[1] * Br[1];
    cval += f1;
    Br += 2;
    Ar += 2;
    k += 2;
  }
  if (k < nVP) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    Br++;
    Ar++;
    k++;
  }
 update:
  f0 = cval * alpha;
  Cr[0] += f0;
}


// this will compute sub-block matrix product: Cij += Aik * Bkj 
static void vpur_transa(double *Cc, const double *Aroot, const double *Bc, double alpha,
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
      inner_loop4_sse(c0, c1, c2, c3, Ac, Br0, Br1, Br2, Br3, alpha, nVP);
      //inner_loop_sse(c0, Ac, Br0, alpha, nVP);
      //inner_loop_sse(c1, Ac, Br1, alpha, nVP);
      //inner_loop_sse(c2, Ac, Br2, alpha, nVP);
      //inner_loop_sse(c3, Ac, Br3, alpha, nVP);
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
      inner_loop2_sse(c0, c1, Ac, Br0, Br1, alpha, nVP);
      //inner_loop_sse(c0, Ac, Br0, alpha, nVP);
      //inner_loop_sse(c1, Ac, Br1, alpha, nVP);
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
      inner_loop_sse(c0, Ac, Br0, alpha, nVP);
      Ac += ldA;
      c0++;
    }
    Cc += ldC;
    Bc += ldB;
  }
    
}

// nP is panel length
void mdata_vpur_unaligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
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

    vpur_transa(Cpy, Acpy, Bcpy, alpha, nC, nA, nB, L-S, E-R, vpL-vpS);

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
void mult_mdata_unaligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
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
      mdata_vpur_unaligned_transa(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}


void mdata_vpur_aligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
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

  // TODO: scaling with beta ....

  while (vpS < nP) {
    nA = vpL - vpS;
    nA += (nA & 0x1);

    Cc = &C->md[S*C->step+R];
    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[R*A->step + vpS];

    vpur_transa(Cc, AvpS, Bc, alpha, C->step, A->step, B->step, L-S, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > nP) {
      vpL = nP;
    }
  }
}


void mult_mdata_aligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
                               double alpha, double beta,
                               int P, int S, int L, int R, int E,
                               int vlen, int NB, int MB)
{
  int i, j, nI, nJ;

  for (j = S; j < L; j += NB) {
    nJ = L - j < NB ? L - j : NB;
    for (i = R; i < E; i += MB) {
      nI = E - i < MB ? E - i : MB;
      mdata_vpur_aligned_transa(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:


