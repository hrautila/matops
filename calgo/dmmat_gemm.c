
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "cmops.h"
#include "matcpy.h"
#include "mult.h"

static inline
int min(int a, int b) {
  return a < b ? a : b;
}

// Scale a tile of M rows by N columns with leading index ldX.
void __SCALE(double *X, int ldX, double beta, int M, int N)
{
  register int i, j;
  if (beta == 1.0) {
    return;
  }

  // set to zero
  if (beta == 0.0) {
    for (j = 0; j < N; j += 4) {
      for (i = 0; i < M; i++) {
        X[i+(j+0)*ldX] = 0.0;
        X[i+(j+1)*ldX] = 0.0;
        X[i+(j+2)*ldX] = 0.0;
        X[i+(j+3)*ldX] = 0.0;
      }
    }
    if (j == N) 
      return;
    for (; j < N; j++) {
      for (i = 0; i < M; i++) {
        X[i+(j+0)*ldX] = 0.0;
      }
    }
    return;
  }
  // scale here
  for (j = 0; j < N; j += 4) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] *= beta;
      X[i+(j+1)*ldX] *= beta;
      X[i+(j+2)*ldX] *= beta;
      X[i+(j+3)*ldX] *= beta;
    }
  }
  if (j == N) 
    return;
  for (; j < N; j++) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] *= beta;
    }
  }
}


// update C block defined by nR rows, nJ columns, nP is A, B common dimension
// A, B data arranged for DOT operations, A matrix is the inner matrix block
// and is loop over nJ times
void __dmult_blk_inner_a(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                         double alpha, int nJ, int nR, int nP)
{
  register int j;

  for (j = 0; j < nJ-3; j += 4) {
    __CMULT4(Cblk, Ablk, Bblk, alpha, j, nR, nP);
  }
  if (j == nJ)
    return;
  // the uneven column stripping part ....
  if (j < nJ-1) {
    __CMULT2(Cblk, Ablk, Bblk, alpha, j, nR, nP);
    j += 2;
  }
  if (j < nJ) {
    __CMULT1(Cblk, Ablk, Bblk, alpha, j, nR, nP);
    j++;
  }
}

// update block of C with A and B panels; A panel is nR*P, B panel is P*nSL
// C block is nR*nSL
void __dmult_inner_a_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                              double alpha, int flags,
                              int P, int nSL, int nRE, 
                              int KB, int NB, int MB, mdata_t *Acpy, mdata_t *Bcpy)
{
  int i, j, k, ip, jp, kp, nP, nI, nJ;
  mdata_t Ca;
  Ca.step = C->step;

  for (jp = 0; jp < nSL; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, nSL-jp);

    for (kp = 0; kp < P; kp += KB) {
      nP = min(KB, P-kp);
    
      if (flags & MTX_TRANSB) {
        __CPTRANS(Bcpy->md, Bcpy->step, &B->md[jp+kp*B->step], B->step, nJ, nP);
      } else {
        __CP(Bcpy->md, Bcpy->step, &B->md[kp+jp*B->step], B->step, nP, nJ);
      }
	  
      for (ip = 0; ip < nRE; ip += MB) {
        nI = min(MB, nRE-ip);
        if (flags & MTX_TRANSA) {
          __CP(Acpy->md, Acpy->step, &A->md[kp+ip*A->step], A->step, nP, nI);
        } else {
          __CPTRANS(Acpy->md, Acpy->step, &A->md[ip+kp*A->step], A->step, nI, nP);
        }
        Ca.md = &C->md[ip+jp*C->step];
      
        for (j = 0; j < nJ-3; j += 4) {
          __CMULT4(&Ca, Acpy, Bcpy, alpha, j, nI, nP);
        }
        if (j == nJ)
          continue;
        // the uneven column stripping part ....
        if (j < nJ-1) {
          __CMULT2(&Ca, Acpy, Bcpy, alpha, j, nI, nP);
          j += 2;
        }
        if (j < nJ) {
          __CMULT1(&Ca, Acpy, Bcpy, alpha, j, nI, nP);
          j++;
        }
      }
    }
  }
}


void __dmult_inner_a_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                             double alpha, double beta, int flags,
                             int P, int S, int L, int R, int E, 
                             int KB, int NB, int MB, mdata_t *Acpy, mdata_t *Bcpy)
{
  int i, j, k, ip, jp, kp, nP, nI, nJ;
  mdata_t Ca;

  Ca.step = C->step;

  // loop over columns of C
  for (jp = S; jp < L; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, L-jp);
	
    // scale C block here.... jp, jp+nJ columns, E-R rows
    __SCALE(&C->md[jp*C->step], C->step, beta, E-R, nJ);

    for (kp = 0; kp < P; kp += KB) {
      nP = min(KB, P-kp);

      if (flags & MTX_TRANSB) {
        __CPTRANS(Bcpy->md, Bcpy->step, &B->md[jp+kp*B->step], B->step, nJ, nP);
      } else {
        __CP(Bcpy->md, Bcpy->step, &B->md[kp+jp*B->step], B->step, nP, nJ);
      }
	  
      for (ip = R; ip < E; ip += MB) {
        nI = min(MB, E-ip);
        if (flags & MTX_TRANSA) {
          __CP(Acpy->md, Acpy->step, &A->md[kp+ip*A->step], A->step, nP, nI);
        } else {
          __CPTRANS(Acpy->md, Acpy->step, &A->md[ip+kp*A->step], A->step, nI, nP);
        }
        Ca.md = &C->md[ip+jp*C->step];

        for (j = 0; j < nJ-3; j += 4) {
          __CMULT4(&Ca, Acpy, Bcpy, alpha, j, min(MB, E-ip), nP);
        }
        if (j == nJ)
          continue;
        // the uneven column stripping part ....
        if (j < nJ-1) {
          __CMULT2(&Ca, Acpy, Bcpy, alpha, j, min(MB, E-ip), nP);
          j += 2;
        }
        if (j < nJ) {
          __CMULT1(&Ca, Acpy, Bcpy, alpha, j, min(MB, E-ip), nP);
          j++;
        }
      }
    }
  }
}


void dmult_mm_blocked4(mdata_t *C, const mdata_t *A, const mdata_t *B,
                       double alpha, double beta, int flags,
                       int P, int S, int L, int R, int E, 
                       int KB, int NB, int MB)
{
  mdata_t Aa, Ba;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Aa.md = Abuf;
  Aa.step = MAX_KB;
  Ba.md = Bbuf;
  Ba.step = MAX_KB;

  if (L-S <= 0 || E-R <= 0) {
    // nothing to do, zero columns or rows
    return;
  }

  // restrict block sizes as data is copied to aligned buffers of
  // predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB  > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  if (alpha == 0.0) {
    __SCALE(&C->md[S*C->step], C->step, beta, L-S, E-R);
    return;
  }
  // update C using A as inner most matrix
  __dmult_inner_a_scale_c(C, A, B, alpha, beta, flags, P, S, L, R, E, KB, NB, MB, &Aa, &Ba);
}


// Local Variables:
// indent-tabs-mode: nil
// End:
