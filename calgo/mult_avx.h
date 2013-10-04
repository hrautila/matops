
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef _MULT_AVX_H
#define _MULT_AVX_H 1

#include <immintrin.h>

#if defined(UNALIGNED)
#define mm_load_A _mm256_loadu_pd
#define mm_load_B _mm256_loadu_pd
#define mm_load   _mm256_loadu_pd
#else
#define mm_load_A _mm256_load_pd
#define mm_load_B _mm256_load_pd
#define mm_load   _mm256_load_pd
#endif

#if defined(UNALIGNED_B)
#undef mm_load_B
#define mm_load_B _mm256_loadu_pd
#endif
#if defined(UNALIGNED_A)
#undef mm_load_A
#define mm_load_A _mm256_loadu_pd
#endif


// update 1x4 block of C; one row, four columns (mult4x1x1)
static inline
void __mult1c4(double *c0, double *c1, double *c2, double *c3,
               const double *a, const double *b0, const double *b1,
               const double *b2, const double *b3, double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, y2, y3, A0, Z, ALPHA;
  y0 = _mm256_set1_pd(0.0);
  y1 = y2 = y3 = y0;

  for (k = 0; k < nR-3; k += 4) {
    A0 = mm_load_A(&a[k]);
    y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
    y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
    y2 += _mm256_mul_pd(A0, mm_load_B(&b2[k]));
    y3 += _mm256_mul_pd(A0, mm_load_B(&b3[k]));
  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  A0 = mm_load_A(&a[k]);
  switch(nR-k) {
  case 3:
    A0 = _mm256_blend_pd(A0, Z, 0x8);
    break;
  case 2:
    A0 = _mm256_blend_pd(A0, Z, 0xC);
    break;
  case 1:
    A0 = _mm256_blend_pd(A0, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
  y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
  y2 += _mm256_mul_pd(A0, mm_load_B(&b2[k]));
  y3 += _mm256_mul_pd(A0, mm_load_B(&b3[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  y2 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y2, y2));
  y3 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y3, y3));
  c0[0] += y0[0] + y0[2];
  c1[0] += y1[0] + y1[2];
  c2[0] += y2[0] + y2[2];
  c3[0] += y3[0] + y3[2];
}


// update 2x4 block of C; two rows, four columns (mult4x2x1)
static inline
void __mult2c4(double *c0, double *c1, double *c2, double *c3,
               const double *a0, const double *a1, const double *b0,
               const double *b1, const double *b2, const double *b3,
               double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, y2, y3, y4, y5, y6, y7, A0, A1, Z, ALPHA;
  y0 = _mm256_set1_pd(0.0);
  y1 = y2 = y3 = y4 = y5 = y6 = y7 = y0;

  for (k = 0; k < nR-3; k += 4) {
    A0 = mm_load_A(&a0[k]);
    A1 = mm_load_A(&a1[k]);
    y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
    y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
    y2 += _mm256_mul_pd(A0, mm_load_B(&b2[k]));
    y3 += _mm256_mul_pd(A0, mm_load_B(&b3[k]));

    y4 += _mm256_mul_pd(A1, mm_load_B(&b0[k]));
    y5 += _mm256_mul_pd(A1, mm_load_B(&b1[k]));
    y6 += _mm256_mul_pd(A1, mm_load_B(&b2[k]));
    y7 += _mm256_mul_pd(A1, mm_load_B(&b3[k]));

  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  A0 = mm_load_A(&a0[k]);
  A1 = mm_load_A(&a1[k]);
  switch(nR-k) {
  case 3:
    A0 = _mm256_blend_pd(A0, Z, 0x8);
    A1 = _mm256_blend_pd(A1, Z, 0x8);
    break;
  case 2:
    A0 = _mm256_blend_pd(A0, Z, 0xC);
    A1 = _mm256_blend_pd(A1, Z, 0xC);
    break;
  case 1:
    A0 = _mm256_blend_pd(A0, Z, 0xE);
    A1 = _mm256_blend_pd(A1, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
  y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
  y2 += _mm256_mul_pd(A0, mm_load_B(&b2[k]));
  y3 += _mm256_mul_pd(A0, mm_load_B(&b3[k]));

  y4 += _mm256_mul_pd(A1, mm_load_B(&b0[k]));
  y5 += _mm256_mul_pd(A1, mm_load_B(&b1[k]));
  y6 += _mm256_mul_pd(A1, mm_load_B(&b2[k]));
  y7 += _mm256_mul_pd(A1, mm_load_B(&b3[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  y2 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y2, y2));
  y3 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y3, y3));
  y4 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y4, y4));
  y5 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y5, y5));
  y6 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y6, y6));
  y7 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y7, y7));
  c0[0] += y0[0] + y0[2];
  c1[0] += y1[0] + y1[2];
  c2[0] += y2[0] + y2[2];
  c3[0] += y3[0] + y3[2];
  c0[1] += y4[0] + y4[2];
  c1[1] += y5[0] + y5[2];
  c2[1] += y6[0] + y6[2];
  c3[1] += y7[0] + y7[2];
}


// update 1x2 block of C; one row, two columns (mult2x1x1)
static inline
void __mult1c2(double *c0, double *c1,
               const double *a, const double *b0, const double *b1,
               double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, A0, ALPHA, Z;

  y0 = _mm256_set1_pd(0.0);
  y1 = y0; 

  for (k = 0; k < nR-3; k += 4) {
    A0 = mm_load_A(&a[k]);
    y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
    y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  A0 = mm_load_A(&a[k]);
  switch(nR-k) {
  case 3:
    A0 = _mm256_blend_pd(A0, Z, 0x8);
    break;
  case 2:
    A0 = _mm256_blend_pd(A0, Z, 0xC);
    break;
  case 1:
    A0 = _mm256_blend_pd(A0, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
  y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  c0[0] += y0[0] + y0[2];
  c1[0] += y1[0] + y1[2];
}


// update 2x2 block of C; (mult2x2x1)
static inline
void __mult2c2(double *c0, double *c1,
               const double *a0, const double *a1,
               const double *b0, const double *b1,
               double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, y2, y3, A0, A1, ALPHA, Z;

  y0 = _mm256_set1_pd(0.0);
  y1 = y0; 
  y2 = y0; 
  y3 = y0; 

  for (k = 0; k < nR-3; k += 4) {
    A0 = mm_load_A(&a0[k]);
    A1 = mm_load_A(&a1[k]);
    y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
    y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
    y2 += _mm256_mul_pd(A1, mm_load_B(&b0[k]));
    y3 += _mm256_mul_pd(A1, mm_load_B(&b1[k]));
  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  A0 = mm_load_A(&a0[k]);
  A1 = mm_load_A(&a1[k]);
  switch(nR-k) {
  case 3:
    A0 = _mm256_blend_pd(A0, Z, 0x8);
    A1 = _mm256_blend_pd(A1, Z, 0x8);
    break;
  case 2:
    A0 = _mm256_blend_pd(A0, Z, 0xC);
    A1 = _mm256_blend_pd(A1, Z, 0xC);
    break;
  case 1:
    A0 = _mm256_blend_pd(A0, Z, 0xE);
    A1 = _mm256_blend_pd(A1, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(A0, mm_load_B(&b0[k]));
  y1 += _mm256_mul_pd(A0, mm_load_B(&b1[k]));
  y2 += _mm256_mul_pd(A1, mm_load_B(&b0[k]));
  y3 += _mm256_mul_pd(A1, mm_load_B(&b1[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  y2 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y2, y2));
  y3 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y3, y3));
  c0[0] += y0[0] + y0[2];
  c1[0] += y1[0] + y1[2];
  c0[1] += y2[0] + y2[2];
  c1[1] += y3[0] + y3[2];
}

// update single element of C; with inner product of A row and B column
static inline
void __mult1c1(double *c, const double *a, const double *b, double alpha, int nR)
{
  register int k;
  register __m256d y0, A, ALPHA, Z;
  y0 = _mm256_set1_pd(0.0);
  for (k = 0; k < nR-3; k += 4) {
    A  = mm_load_A(&a[k]);
    y0 += _mm256_mul_pd(A, mm_load_B(&b[k]));
  }
  if (k == nR)
    goto update;

  Z = _mm256_set1_pd(0.0);
  A = mm_load_A(&a[k]);
  switch(nR-k) {
  case 3:
    A = _mm256_blend_pd(A, Z, 0x8);
    break;
  case 2:
    A = _mm256_blend_pd(A, Z, 0xC);
    break;
  case 1:
    A = _mm256_blend_pd(A, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(A, mm_load_B(&b[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  c[0] += y0[0] + y0[2];
}


// version for breadth-first update of C; update C row-wise
// 1 to 4 rows of A, 1 or 2 rows of B, update atmost 4x2 block of C 

// update 4x1 block of C; four rows, one column (dmult4x1x1)
static inline
void __mult4c1(double *c0, 
               const double *a0, const double *a1,
               const double *a2, const double *a3,
               const double *b0, double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, y2, y3, B0, Z, ALPHA;
  y0 = _mm256_set1_pd(0.0);
  y1 = y2 = y3 = y0;

  for (k = 0; k < nR-3; k += 4) {
    B0 = mm_load_B(&b0[k]);
    y0 += _mm256_mul_pd(B0, mm_load_A(&a0[k]));
    y1 += _mm256_mul_pd(B0, mm_load_A(&a1[k]));
    y2 += _mm256_mul_pd(B0, mm_load_A(&a2[k]));
    y3 += _mm256_mul_pd(B0, mm_load_A(&a3[k]));
  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  B0 = mm_load_B(&b0[k]);
  switch(nR-k) {
  case 3:
    B0 = _mm256_blend_pd(B0, Z, 0x8);
    break;
  case 2:
    B0 = _mm256_blend_pd(B0, Z, 0xC);
    break;
  case 1:
    B0 = _mm256_blend_pd(B0, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(B0, mm_load_A(&a0[k]));
  y1 += _mm256_mul_pd(B0, mm_load_A(&a1[k]));
  y2 += _mm256_mul_pd(B0, mm_load_A(&a2[k]));
  y3 += _mm256_mul_pd(B0, mm_load_A(&a3[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  y2 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y2, y2));
  y3 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y3, y3));
  c0[0] += y0[0] + y0[2];
  c0[1] += y1[0] + y1[2];
  c0[2] += y2[0] + y2[2];
  c0[3] += y3[0] + y3[2];
}


// update 4x2 block of C; four rows, one column (dmult4x2x1)
static inline
void __mult4c2(double *c0, double *c1,
               const double *a0, const double *a1,
               const double *a2, const double *a3,
               const double *b0, const double *b1, double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, y2, y3, y4, y5, y6, y7, B0, B1, Z, ALPHA;
  y0 = _mm256_set1_pd(0.0);
  y1 = y2 = y3 = y0;
  y4 = y5 = y6 = y7 = y0;

  for (k = 0; k < nR-3; k += 4) {
    B0 = mm_load_B(&b0[k]);
    B1 = mm_load_B(&b1[k]);
    y0 += _mm256_mul_pd(B0, mm_load_A(&a0[k]));
    y1 += _mm256_mul_pd(B0, mm_load_A(&a1[k]));
    y2 += _mm256_mul_pd(B0, mm_load_A(&a2[k]));
    y3 += _mm256_mul_pd(B0, mm_load_A(&a3[k]));
    y4 += _mm256_mul_pd(B1, mm_load_A(&a0[k]));
    y5 += _mm256_mul_pd(B1, mm_load_A(&a1[k]));
    y6 += _mm256_mul_pd(B1, mm_load_A(&a2[k]));
    y7 += _mm256_mul_pd(B1, mm_load_A(&a3[k]));

  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  B0 = mm_load_B(&b0[k]);
  B1 = mm_load_B(&b1[k]);
  switch(nR-k) {
  case 3:
    B0 = _mm256_blend_pd(B0, Z, 0x8);
    B1 = _mm256_blend_pd(B1, Z, 0x8);
    break;
  case 2:
    B0 = _mm256_blend_pd(B0, Z, 0xC);
    B1 = _mm256_blend_pd(B1, Z, 0xC);
    break;
  case 1:
    B0 = _mm256_blend_pd(B0, Z, 0xE);
    B1 = _mm256_blend_pd(B1, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(B0, mm_load_A(&a0[k]));
  y1 += _mm256_mul_pd(B0, mm_load_A(&a1[k]));
  y2 += _mm256_mul_pd(B0, mm_load_A(&a2[k]));
  y3 += _mm256_mul_pd(B0, mm_load_A(&a3[k]));
  y4 += _mm256_mul_pd(B1, mm_load_A(&a0[k]));
  y5 += _mm256_mul_pd(B1, mm_load_A(&a1[k]));
  y6 += _mm256_mul_pd(B1, mm_load_A(&a2[k]));
  y7 += _mm256_mul_pd(B1, mm_load_A(&a3[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  y2 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y2, y2));
  y3 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y3, y3));
  y4 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y4, y4));
  y5 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y5, y5));
  y6 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y6, y6));
  y7 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y7, y7));
  c0[0] += y0[0] + y0[2];
  c0[1] += y1[0] + y1[2];
  c0[2] += y2[0] + y2[2];
  c0[3] += y3[0] + y3[2];
  c1[0] += y4[0] + y4[2];
  c1[1] += y5[0] + y5[2];
  c1[2] += y6[0] + y6[2];
  c1[3] += y7[0] + y7[2];
}



// update 2x1 block of C; two rows, one column; (dmult2x1x1)
static inline
void __mult2c1(double *c0, 
               const double *a0, const double *a1,
               const double *b0, double alpha, int nR)
{
  register int k;
  register __m256d y0, y1, B0, ALPHA, Z;

  y0 = _mm256_set1_pd(0.0);
  y1 = y0; 

  for (k = 0; k < nR-3; k += 4) {
    B0 = mm_load_B(&b0[k]);
    y0 += _mm256_mul_pd(B0, mm_load_A(&a0[k]));
    y1 += _mm256_mul_pd(B0, mm_load_A(&a1[k]));
  }
  if (k == nR)
    goto update;

  Z  = _mm256_set1_pd(0.0);
  B0 = mm_load_B(&b0[k]);
  switch(nR-k) {
  case 3:
    B0 = _mm256_blend_pd(B0, Z, 0x8);
    break;
  case 2:
    B0 = _mm256_blend_pd(B0, Z, 0xC);
    break;
  case 1:
    B0 = _mm256_blend_pd(B0, Z, 0xE);
    break;
  }
  y0 += _mm256_mul_pd(B0, mm_load_A(&a0[k]));
  y1 += _mm256_mul_pd(B0, mm_load_A(&a1[k]));

update:
  ALPHA = _mm256_set1_pd(alpha);
  y0 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y0, y0));
  y1 = _mm256_mul_pd(ALPHA, _mm256_hadd_pd(y1, y1));
  c0[0] += y0[0] + y0[2];
  c0[1] += y1[0] + y1[2];
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:

