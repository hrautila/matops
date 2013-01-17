
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <x86intrin.h>

#include "cmops.h"

// max values of block sizes for unaligned cases.
#define MAX_UA_MB 128
#define MAX_UA_NB 64
#define MAX_UA_VP 64

#define MAX_MB 256
#define MAX_NB 256
#define MAX_VP 192

// This will do efectively AXPY C[:,i] = w * A[:,k] + C[:,i] where w = alpha * B[k,i] 
static void inner_mult(double *Cr, const double *Ar, const double *Br, double alpha, int m) {
  register int i;
  register double c0, c1, cf;
  cf = Br[0] * alpha;
  for (i = 0; i < m-3; i += 4) {
    c0 = Ar[0] * cf;
    Cr[0] += c0;
    c1 = Ar[1] * cf;
    Cr[1] += c1;
    Cr += 2;
    Ar += 2;
    c0 = Ar[0] * cf;
    Cr[0] += c0;
    c1 = Ar[1] * cf;
    Cr[1] += c1;
    Cr += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    c0 = Ar[0] * cf;
    Cr[0] += c0;
    c1 = Ar[1] * cf;
    Cr[1] += c1;
    Cr += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    c0 = Ar[0] * cf;
    Cr[0] += c0;
  }
}

// This will do efectively AXPY C[:,i] = w * A[:,k] + C[:,i] where w = alpha * B[k,i] 
static inline void inner_mult_sse(double *Cr, const double *Ar, const double *Br, double alpha, int m) {
  register int i;
  register __m128d Av, Bv, Cv, Tv, Al;
  Al = _mm_set1_pd(alpha);
  Bv = _mm_set1_pd(Br[0]);
  Bv = Bv * Al;
  for (i = 0; i < m-3; i += 4) {
    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
    Cr += 2;
    Ar += 2;

    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
    Cr += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    // next 2
    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
    Cr += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    // the last one and the extra for odd case. 
    Av =  _mm_load_pd(Ar);
    Tv = _mm_mul_pd(Av, Bv);
    Cv =  _mm_load_pd(Cr);
    Cv = _mm_add_pd(Cv, Tv);
    _mm_store_pd(Cr, Cv);
  }
}

static void inner_mult4_sse(double *c0, double *c1, double *c2, double *c3, const double *Ar,
                            const double *b0, const double *b1, const double *b2, const double *b3,
                            double alpha, int m) {
  register int i;
  register __m128d T0, T1, Av;
  register __m128d C0, C1, B0, B1, B2, B3;

  Av = _mm_set1_pd(alpha);
  B0 = _mm_set1_pd(b0[0]);
  B1 = _mm_set1_pd(b1[0]);
  B2 = _mm_set1_pd(b2[0]);
  B3 = _mm_set1_pd(b3[0]);
  B0 = B0 * Av;
  B1 = B1 * Av;
  B2 = B2 * Av;
  B3 = B3 * Av;
  for (i = 0; i < m-3; i += 4) {
    Av = _mm_load_pd(Ar);

    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T0 = Av * B2;
    C0 = _mm_load_pd(c2);
    C0 = C0 + T0;
    _mm_store_pd(c2, C0);

    T1 = Av * B3;
    C1 = _mm_load_pd(c3);
    C1 = C1 + T1;
    _mm_store_pd(c3, C1);

    c0 += 2;
    c1 += 2;
    c2 += 2;
    c3 += 2;
    Ar += 2;

    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T0 = Av * B2;
    C0 = _mm_load_pd(c2);
    C0 = C0 + T0;
    _mm_store_pd(c2, C0);

    T1 = Av * B3;
    C1 = _mm_load_pd(c3);
    C1 = C1 + T1;
    _mm_store_pd(c3, C1);

    c0 += 2;
    c1 += 2;
    c2 += 2;
    c3 += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    // next 2
    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T0 = Av * B2;
    C0 = _mm_load_pd(c2);
    C0 = C0 + T0;
    _mm_store_pd(c2, C0);

    T1 = Av * B3;
    C1 = _mm_load_pd(c3);
    C1 = C1 + T1;
    _mm_store_pd(c3, C1);

    c0 += 2;
    c1 += 2;
    c2 += 2;
    c3 += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    // the last one and the extra for odd case. 
    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    T0 = Av * B2;
    C0 = _mm_load_pd(c2);
    C0 = C0 + T0;
    _mm_store_pd(c2, C0);

    T1 = Av * B3;
    C1 = _mm_load_pd(c3);
    C1 = C1 + T1;
    _mm_store_pd(c3, C1);
  }
}

static void inner_mult2_sse(double *c0, double *c1, const double *Ar,
                            const double *b0, const double *b1, double alpha, int m)
{
  register int i;
  register __m128d T0, T1, Av;
  register __m128d C0, C1, B0, B1, B2, B3;

  Av = _mm_set1_pd(alpha);
  B0 = _mm_set1_pd(b0[0]);
  B1 = _mm_set1_pd(b1[0]);
  B0 = B0 * Av;
  B1 = B1 * Av;
  for (i = 0; i < m-3; i += 4) {
    Av = _mm_load_pd(Ar);

    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    c0 += 2;
    c1 += 2;
    Ar += 2;

    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    c0 += 2;
    c1 += 2;
    Ar += 2;
  }
  if (i == m)
    return;

  if (i < m-1) {
    // next 2
    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);

    c0 += 2;
    c1 += 2;
    Ar += 2;
    i += 2;
  }

  if (i < m) {
    // the last one and the extra for odd case. 
    Av =  _mm_load_pd(Ar);
    T0 = Av * B0;
    C0 = _mm_load_pd(c0);
    C0 = C0 + T0;
    _mm_store_pd(c0, C0);

    T1 = Av * B1;
    C1 = _mm_load_pd(c1);
    C1 = C1 + T1;
    _mm_store_pd(c1, C1);
  }
}


// this will compute sub-block matrix product: Cij += Aik * Bkj 
static void vpur_notrans(double *Cc, const double *Aroot, const double *Bc, double alpha,
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
      inner_mult4_sse(c0, c1, c2, c3, Ac, Br0, Br1, Br2, Br3, alpha, nRE);
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
      inner_mult2_sse(c0, c1, Ac, Br0, Br1, alpha, nRE);
      //inner_mult_sse(c0, Ac, Br0, alpha, nRE);
      //inner_mult_sse(c1, Ac, Br1, alpha, nRE);
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
      inner_mult_sse(c0, Ac, Br0,  alpha, nRE);
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
void mdata_vpur_aligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                double alpha, double beta,
                                int P, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL;
  const double *Bc, *Ac, *AvpS;
  double *Cc;

  vpS = 0;
  vpL = vlen < P ? vlen : P;

  // TODO: scaling with beta ...

  while (vpS < P) {
    // block start C[R, S]
    Cc = &C->md[S*C->step+R];
    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    vpur_notrans(Cc, AvpS, Bc, alpha, C->step, A->step, B->step, L-S, E-R, vpL-vpS);

    vpS = vpL;
    vpL += vlen;
    if (vpL > P) {
      vpL = P;
    }
  }
}


// Use this when rows of C and A are aligned to 16bytes, ie C and A row strides
// are even.
void mult_mdata_aligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, double beta,
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
      mdata_vpur_aligned_notrans(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}

// nP is panel length
void mdata_vpur_unaligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B, double alpha, double beta,
                                  int nP, int S, int L, int R, int E, int vlen)
{
  int j, k, vpS, vpL, nC, nB;
  const double *Bc, *Ac, *AvpS;
  const double *Br0, *Br1, *Br2, *Br3;
  double *Cc, *c0, *c1, *c2, *c3;
  double Cpy[MAX_UA_NB*MAX_UA_MB]  __attribute__((aligned(16)));
  double Acpy[MAX_UA_VP*MAX_UA_MB] __attribute__((aligned(16)));


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
    // column viewport start in panel B[:,S]
    Bc = &B->md[S*B->step + vpS];
    // row viewport start A[R,:]
    AvpS = &A->md[vpS*A->step + R];

    colcpy(Acpy, nC, AvpS, A->step, E-R, vpL-vpS);

    vpur_notrans(Cpy, Acpy, Bc, alpha, nC, nC, B->step, L-S, E-R, vpL-vpS);

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
void mult_mdata_unaligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
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
      mdata_vpur_unaligned_notrans(C, A, B, alpha, beta, P, j, j+nJ, i, i+nI, vlen);
    }
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:
