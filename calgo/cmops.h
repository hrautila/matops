
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef _CMOPS_H
#define _CMOPS_H

#include <stdio.h>

// max values of block sizes for unaligned cases.
#define MAX_UA_MB 128
#define MAX_UA_NB 64
#define MAX_UA_VP 64

#define MAX_MB 256
#define MAX_NB 256
#define MAX_VP 192

#define OFFSET(a,b) ((unsigned int)(a-b))

// simple structure to hold column-major matrix data; 'md' points to first element
// in matrix at index (0, 0); 'step' is the row stride for data i.e. how many elements 
// between A[i,j] and A[i,j+1]
typedef struct mdata {
  double *md;
  int step;
} mdata_t;

extern void *memcpy(void *, const void *, size_t);

extern inline void colcpy(double *dst, int nD, const double *src, int nS, int nL, int nC) {
  register int i;
  for (i = 0; i < nC; i++) {
    memcpy(dst, src, nL*sizeof(double));
    dst += nD;
    src += nS;
  }
}


// for data, C and A, aligned at 16 bytes.
extern void mult_mdata_aligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
				       double alpha, double beta,
				       int P, int S, int L, int R, int E,
				       int vlen, int NB, int MB);

// for data not aligned at 16 bytes.
extern void mult_mdata_unaligned_notrans(mdata_t *C, const mdata_t *A, const mdata_t *B,
					 double alpha, double beta, 
					 int P, int S, int L, int R, int E,
					 int vlen, int NB, int MB);

// for data not aligned at 16 bytes.
extern void mult_mdata_unaligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
					double alpha, double beta, 
					int P, int S, int L, int R, int E,
					int vlen, int NB, int MB);

extern void mult_mdata_aligned_transa(mdata_t *C, const mdata_t *A, const mdata_t *B,
				      double alpha, double beta, 
				      int P, int S, int L, int R, int E,
				      int vlen, int NB, int MB);


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
