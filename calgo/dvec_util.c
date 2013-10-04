
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "cmops.h"

/*
 * Z[0] = alpha*(X*Y.T) + beta*Z[0]
 */
void dvec_dots(mvec_t *Z, const mvec_t *X,  const mvec_t *Y, double alpha, double beta, int N)
{
  register int i, kx, ky;
  register double c0, c1, c2, c3, x0, x1, x2, x3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    c0 += x0*Y->md[(i+0)*Y->inc];
    c1 += x1*Y->md[(i+1)*Y->inc];
    c2 += x2*Y->md[(i+2)*Y->inc];
    c3 += x3*Y->md[(i+3)*Y->inc];
  }    
  if (i == N)
    goto update;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    c0 += X->md[kx] * Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 2:
    c1 += X->md[kx] * Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 1:
    c2 += X->md[kx] * Y->md[ky];
  }
 update:
  Z->md[0] *= beta;
  Z->md[0] += alpha * (c0 + c1 + c2 + c3);
}

/*
 * return: alpha*(X*Y.T)
 */
double dvec_dot(const mvec_t *X,  const mvec_t *Y, double alpha, int N)
{
  register int i, kx, ky;
  register double c0, c1, c2, c3, x0, x1, x2, x3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    c0 += x0*Y->md[(i+0)*Y->inc];
    c1 += x1*Y->md[(i+1)*Y->inc];
    c2 += x2*Y->md[(i+2)*Y->inc];
    c3 += x3*Y->md[(i+3)*Y->inc];
  }    
  if (i == N)
    goto update;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    c0 += X->md[kx] * Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 2:
    c1 += X->md[kx] * Y->md[ky];
    kx += X->inc; ky += Y->inc;
  case 1:
    c2 += X->md[kx] * Y->md[ky];
  }
 update:
  return alpha * (c0 + c1 + c2 + c3);
}

/*
 * Y = alpha*X + Y
 */
void dvec_axpy(mvec_t *Y,  const mvec_t *X, double alpha, int N)
{
  register int i, kx, ky;
  register double y0, y1, y2, y3, x0, x1, x2, x3;

  // gcc uses different XMM target registers for yN, xN; 
  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    y0 += alpha*x0;
    y1 += alpha*x1;
    y2 += alpha*x2;
    y3 += alpha*x3;
    Y->md[(i+0)*Y->inc] = y0;
    Y->md[(i+1)*Y->inc] = y1;
    Y->md[(i+2)*Y->inc] = y2;
    Y->md[(i+3)*Y->inc] = y3;
  }    
  if (i == N)
	return;

  kx = i*X->inc; ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
  }
}

/*
 * return ||X - Y||_2
 */
double dvec_diff_nrm2(const mvec_t *X,  const mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double c0, c1, c2, c3, y0, y1, y2, y3, x0, x1, x2, x3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    x0 = x0 - y0;
    x1 = x1 - y1;
    x2 = x2 - y2;
    x3 = x3 - y3;
    x0 = fabs(x0);
    x1 = fabs(x1);
    x2 = fabs(x2);
    x3 = fabs(x3);
    c0 += x0 * x0;
    c1 += x1 * x1;
    c2 += x2 * x2;
    c3 += x3 * x3;
  }    
  if (i == N)
    goto update;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    x0 = fabs(X->md[kx] - Y->md[ky]);
    c0 += x0 * x0;
    kx += X->inc; ky += Y->inc;
  case 2:
    x1 = fabs(X->md[kx] - Y->md[ky]);
    c1 += x1 * x1;
    kx += X->inc; ky += Y->inc;
  case 1:
    x2 = fabs(X->md[kx] - Y->md[ky]);
    c2 += x2 * x2;
  }
 update:
  return sqrt(c0 + c1 + c2 + c3);
}

// return vector norm 
double dvec_nrm2(const mvec_t *X,  int N)
{
  register int i, k;
  register double c0, c1, c2, c3, a0, a1, a2, a3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    a0 = X->md[(i+0)*X->inc];
    a1 = X->md[(i+1)*X->inc];
    a2 = X->md[(i+2)*X->inc];
    a3 = X->md[(i+3)*X->inc];
    a0 = fabs(a0);
    a1 = fabs(a1);
    a2 = fabs(a2);
    a3 = fabs(a3);
    c0 += a0*a0;
    c1 += a1*a1;
    c2 += a2*a2;
    c3 += a3*a3;
  }    
  if (i == N)
    goto update;

  k = i*X->inc;
  switch (N-i) {
  case 3:
    a0 = fabs(X->md[k]);
    c0 += a0*a0;
    k += X->inc;
  case 2:
    a1 = fabs(X->md[k]);
    c1 += a1*a1;
    k += X->inc;
  case 1:
    a2 = fabs(X->md[k]);
    c2 += a2*a2;
  }
 update:
  return sqrt(c0 + c1 + c2 + c3);
}

// return sum of absolute values
double dvec_asum(const mvec_t *X,  int N)
{
  register int i, k;
  register double c0, c1, c2, c3, a0, a1, a2, a3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    a0 = X->md[(i+0)*X->inc];
    a1 = X->md[(i+1)*X->inc];
    a2 = X->md[(i+2)*X->inc];
    a3 = X->md[(i+3)*X->inc];
    a0 = fabs(a0);
    a1 = fabs(a1);
    a2 = fabs(a2);
    a3 = fabs(a3);
    c0 += a0;
    c1 += a1;
    c2 += a2;
    c3 += a3;
  }    
  if (i == N)
    goto update;
  
  k = i*X->inc;
  switch (N-i) {
  case 3:
    c0 += fabs(X->md[k]);
    k += X->inc;
  case 2:
    c1 += fabs(X->md[k]);
    k += X->inc;
  case 1:
    c2 += fabs(X->md[k]);
  }
 update:
  return c0 + c1 + c2 + c3;
}

// return index of max absolute value
int dvec_iamax(const mvec_t *X,  int N)
{
  register int i, ix, n;
  register double max, c0, c1;

  if (N <= 1)
    return 0;

  max = 0.0;
  ix = 0;
  for (i = 0; i < N-1; i += 2) {
    c0 = fabs(X->md[(i+0)*X->inc]);
    c1 = fabs(X->md[(i+1)*X->inc]);
    if (c1 > c0) {
      n = 1;
      c0 = c1;
    }
    if (c0 > max) {
      ix = i+n;
      max = c0;
    }
    n = 0;
  }    
  if (i < N) {
    c0 = fabs(X->md[i*X->inc]);
    ix = c0 > max ? N-1 : ix;
  }
  return ix;
}

/*
 * X <--> Y
 */
void dvec_swap(mvec_t *X,  mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double y0, y1, y2, y3, x0, x1, x2, x3;

  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    X->md[(i+0)*X->inc] = y0;
    X->md[(i+1)*X->inc] = y1;
    X->md[(i+2)*X->inc] = y2;
    X->md[(i+3)*X->inc] = y3;
    Y->md[(i+0)*Y->inc] = x0;
    Y->md[(i+1)*Y->inc] = x1;
    Y->md[(i+2)*Y->inc] = x2;
    Y->md[(i+3)*Y->inc] = x3;
  }    
  if (i == N)
    return;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
  }
}

/*
 * X = X/alpha
 */
void dvec_invscal(mvec_t *X,  double alpha, int N)
{
  register int i, k;
  register double f0, f1, f2, f3;
  register double *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] / alpha;
    f1 =  X->md[(i+1)*X->inc] / alpha;
    f2 =  X->md[(i+2)*X->inc] / alpha;
    f3 =  X->md[(i+3)*X->inc] / alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] /= alpha;
    k += X->inc;
  case 2:
    x0[k] /= alpha;
    k += X->inc;
  case 1:
    x0[k] /= alpha;
  }
}

/*
 * X = alpha*X
 */
void dvec_scal(mvec_t *X,  double alpha, int N)
{
  register int i, k;
  register double f0, f1, f2, f3;
  register double *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] * alpha;
    f1 =  X->md[(i+1)*X->inc] * alpha;
    f2 =  X->md[(i+2)*X->inc] * alpha;
    f3 =  X->md[(i+3)*X->inc] * alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] *= alpha;
    k += X->inc;
  case 2:
    x0[k] *= alpha;
    k += X->inc;
  case 1:
    x0[k] *= alpha;
  }
}

/*
 * X := Y
 */
void dvec_copy(mvec_t *X,  mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double f0, f1, f2, f3;

  // gcc compiles loop body to use different target XMM registers
  for (i = 0; i < N-3; i += 4) {
    f0 = Y->md[(i+0)*Y->inc];
    f1 = Y->md[(i+1)*Y->inc];
    f2 = Y->md[(i+2)*Y->inc];
    f3 = Y->md[(i+3)*Y->inc];
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // calculate indexes only once
  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    X->md[kx] = Y->md[ky];
    kx++; ky++;
  case 2:
    X->md[kx] = Y->md[ky];
    kx++; ky++;
  case 1:
    X->md[kx] = Y->md[ky];
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
