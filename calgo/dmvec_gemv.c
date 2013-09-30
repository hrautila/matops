
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "cmops.h"


static inline
void __vmult1axpy(double *y, int incy,
                 const double *a0, const double *x, int incx,
                 double alpha, int nR)
{
  register int k;
  register double cf;

  cf = alpha*x[0];

  for (k = 0; k < nR-3; k += 4) {
    y[(k+0)*incy] += a0[k+0]*cf;
    y[(k+1)*incy] += a0[k+1]*cf;
    y[(k+2)*incy] += a0[k+2]*cf;
    y[(k+3)*incy] += a0[k+3]*cf;
  }
  if (k == nR)
    return;

  switch (nR-k) {
  case 3:
    y[(k+0)*incy] += a0[k+0]*cf;
    k++;
  case 2:
    y[(k+0)*incy] += a0[k+0]*cf;
    k++;
  case 1:
    y[(k+0)*incy] += a0[k+0]*cf;
  }
}


static inline
void __vmult2axpy(double *y, int incy,
                 const double *a0, const double *a1, const double *x, int incx,
                 double alpha, int nR)
{
  register int k;
  register double cf0, cf1, t0, t1, t2, t3, t4, t5, t6, t7;

  cf0 = alpha*x[0];
  cf1 = alpha*x[incx];

  for (k = 0; k < nR-3; k += 4) {
    t0 = a0[k+0]*cf0;
    t1 = a0[k+1]*cf0;
    t2 = a0[k+2]*cf0;
    t3 = a0[k+3]*cf0;
    t4 = a1[k+0]*cf1;
    t5 = a1[k+1]*cf1;
    t6 = a1[k+2]*cf1;
    t7 = a1[k+3]*cf1;
    y[(k+0)*incy] += t0 + t4;
    y[(k+1)*incy] += t1 + t5;
    y[(k+2)*incy] += t2 + t6;
    y[(k+3)*incy] += t3 + t7;
  }
  if (k == nR)
    return;

  switch (nR-k) {
  case 3:
    y[k*incy] += a0[k]*cf0 + a1[k]*cf1;
    k++;
  case 2:
    y[k*incy] += a0[k]*cf0 + a1[k]*cf1;
    k++;
  case 1:
    y[k*incy] += a0[k]*cf0 + a1[k]*cf1;
  }
}


static inline
void __vmult4axpy(double *y, int incy,
                 const double *a0, const double *a1, const double *a2, const double *a3,
                 const double *x, int incx,
                 double alpha, int nR)
{
  register int k;
  register double cf0, cf1, cf2, cf3, t0, t1, t2, t3, t4, t5, t6, t7;

  cf0 = alpha*x[0];
  cf1 = alpha*x[incx];
  cf2 = alpha*x[2*incx];
  cf3 = alpha*x[3*incx];

  for (k = 0; k < nR-1; k += 2) {
    t0 = a0[k+0]*cf0;
    t1 = a0[k+1]*cf0;

    t2 = a1[k+0]*cf1;
    t3 = a1[k+1]*cf1;

    t4 = a2[k+0]*cf2;
    t5 = a2[k+1]*cf2;

    t6 = a3[k+0]*cf3;
    t7 = a3[k+1]*cf3;

    y[(k+0)*incy] += t0 + t2 + t4 + t6;
    y[(k+1)*incy] += t1 + t3 + t5 + t7;
  }
  if (k == nR)
    return;

  t0 = a0[k+0]*cf0;
  t2 = a1[k+0]*cf1;
  t4 = a2[k+0]*cf2;
  t6 = a3[k+0]*cf3;
  y[k*incy] += t0 + t2 + t4 + t6;
}



static inline
void __vmult1dot(double *y, int incy,
                const double *a0, const double *x, int incx,
                double alpha, int nR)
{
  register int k;
  register double t0, t1, t2, t3;

  t0 = t1 = t2 = t3 = 0.0;
  for (k = 0; k < nR-3; k += 4) {
    t0 += a0[k+0]*x[(k+0)*incx];
    t1 += a0[k+1]*x[(k+1)*incx];
    t2 += a0[k+2]*x[(k+2)*incx];
    t3 += a0[k+3]*x[(k+3)*incx];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    t0 += a0[k]*x[k*incx];
    k++;
  case 2:
    t1 += a0[k]*x[k*incx];
    k++;
  case 1:
    t2 += a0[k]*x[k*incx];
  }
 update:
  t0 += t1; t2 += t3;
  y[0]    += (t0 + t2)*alpha;
}


static inline
void __vmult2dot(double *y, int incy,
                const double *a0, const double *a1, const double *x, int incx,
                double alpha, int nR)
{
  register int k;
  register double t0, t1, t2, t3, t4, t5, t6, t7;

  t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0.0;
  for (k = 0; k < nR-3; k += 4) {
    t0 += a0[k+0]*x[(k+0)*incx];
    t1 += a0[k+1]*x[(k+1)*incx];
    t2 += a0[k+2]*x[(k+2)*incx];
    t3 += a0[k+3]*x[(k+3)*incx];

    t4 += a1[k+0]*x[(k+0)*incx];
    t5 += a1[k+1]*x[(k+1)*incx];
    t6 += a1[k+2]*x[(k+2)*incx];
    t7 += a1[k+3]*x[(k+3)*incx];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    t0 += a0[k]*x[k*incx];
    t4 += a1[k]*x[k*incx];
    k++;
  case 2:
    t1 += a0[k]*x[k*incx];
    t5 += a1[k]*x[k*incx];
    k++;
  case 1:
    t2 += a0[k]*x[k*incx];
    t6 += a1[k]*x[k*incx];
  }
 update:
  t0 += t1; t2 += t3;
  t4 += t5; t6 += t7;
  y[0]    += (t0 + t2)*alpha;
  y[incy] += (t4 + t6)*alpha;
}

static inline
void __vmult4dot(double *y, int incy,
                const double *a0, const double *a1, const double *a2, const double *a3,
                const double *x, int incx,
                double alpha, int nR)
{
  register int k;
  register double t0, t1, t2, t3, t4, t5, t6, t7;

  t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0.0;
  for (k = 0; k < nR-1; k += 2) {
    t0 += a0[k+0]*x[(k+0)*incx];
    t1 += a0[k+1]*x[(k+1)*incx];

    t2 += a1[k+0]*x[(k+0)*incx];
    t3 += a1[k+1]*x[(k+1)*incx];

    t4 += a2[k+0]*x[(k+0)*incx];
    t5 += a2[k+1]*x[(k+1)*incx];

    t6 += a3[k+0]*x[(k+0)*incx];
    t7 += a3[k+1]*x[(k+1)*incx];
  }
  if (k == nR)
    goto update;

  t0 += a0[k]*x[k*incx];
  t2 += a1[k]*x[k*incx];
  t4 += a2[k]*x[k*incx];
  t6 += a3[k]*x[k*incx];

 update:
  t0 += t1; t2 += t3;
  t4 += t5; t6 += t7;
  y[0]      += t0*alpha;
  y[incy]   += t2*alpha;
  y[2*incy] += t4*alpha;
  y[3*incy] += t6*alpha;
}


// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
void dmult_gemv2(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                 double alpha, double beta, int flags,
                 int S, int L, int R, int E)
{
  int i, j;
  register double *y;
  register const double *x;
  register const double *a0, *a1, *a2, *a3;

  // L - S is columns in A, elements in X
  // E - R is rows in A, elements in Y 
  if (L - S <= 0 || E - R <= 0) {
    return;
  }

  if (beta != 1.0) {
    if (beta != 0.0) {
      for (i = R; i < E; i++) {
        Y->md[i*Y->inc] *= beta;
      }
    } else {
      for (i = R; i < E; i++) {
        Y->md[i*Y->inc] = 0.0;
      }
    }
  }

  if ((flags & MTX_TRANSA) || (flags & MTX_TRANS)) {

    x = &X->md[S*X->inc];
    for (i = R; i < E-3; i += 4) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[(i+0)*A->step];
      a1 = &A->md[(i+1)*A->step];
      a2 = &A->md[(i+2)*A->step];
      a3 = &A->md[(i+3)*A->step];
      __vmult4dot(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, L-S);
    }
    if (i == E)
      return;

    switch (E-i) {
    case 3:
    case 2:
      y = &Y->md[i*Y->inc];
      a0 = &A->md[(i+0)*A->step];
      a1 = &A->md[(i+1)*A->step];
      __vmult2dot(y, Y->inc, a0, a1, x, X->inc, alpha, L-S);
      i += 2;
    }
    if (i < E) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[(i+0)*A->step];
      __vmult1dot(y, Y->inc, a0, x, X->inc, alpha, L-S);
    }
    return;
  }

  // Non-Transposed A here


  y = &Y->md[R*Y->inc];
  for (j = S; j < L-3; j += 4) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    a2 = &A->md[R+(j+2)*A->step];
    a3 = &A->md[R+(j+3)*A->step];
    __vmult4axpy(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, E-R);
  }

  if (j == L)
    return;

  switch (L-j) {
  case 3:
  case 2:
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    __vmult2axpy(y, Y->inc, a0, a1, x, X->inc, alpha, E-R);
    j += 2;
  }
  if (j < L) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    __vmult1axpy(y, Y->inc, a0, x, X->inc, alpha, E-R);
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:
