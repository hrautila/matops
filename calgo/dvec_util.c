
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "cmops.h"

void dvec_dots(mvec_t *Z, const mvec_t *X,  const mvec_t *Y, double alpha, double beta, int N)
{
  register int i;
  register double c0, c1;
  register const double *Xc, *Yc;

  Xc = X->md;
  Yc = Y->md;
  c0 = 0.0; c1 = 0.0;
  for (i = 0; i < N-1; i += 2) {
    c0 += Xc[0] * Yc[0];
    Xc += X->inc;
    Yc += Y->inc;
    c1 += Xc[0] * Yc[0];
    Xc += X->inc;
    Yc += Y->inc;
  }    
  if (i < N) {
    c0 += Xc[0] * Yc[0];
  }
  Z->md[0] *= beta;
  Z->md[0] += alpha * (c0 + c1);
}

double dvec_dot(const mvec_t *X,  const mvec_t *Y, double alpha, int N)
{
  register int i;
  register double c0, c1;
  register const double *Xc, *Yc;

  Xc = X->md;
  Yc = Y->md;
  c0 = 0.0; c1 = 0.0;
  for (i = 0; i < N-1; i += 2) {
    c0 += Xc[0] * Yc[0];
    Xc += X->inc;
    Yc += Y->inc;
    c1 += Xc[0] * Yc[0];
    Xc += X->inc;
    Yc += Y->inc;
  }    
  if (i < N) {
    c0 += Xc[0] * Yc[0];
  }
  return alpha * (c0 + c1);
}

double dvec_nrm2(const mvec_t *X,  const mvec_t *Y, int N)
{
  register int i;
  register double c0, c1, d0, d1;
  register const double *Xc, *Yc;

  Xc = X->md;
  Yc = Y->md;
  c0 = 0.0; c1 = 0.0;
  for (i = 0; i < N-1; i += 2) {
    d0 = Xc[0] - Yc[0];
    c0 += d0 * d0;
    Xc += X->inc;
    Yc += Y->inc;

    d1 = Xc[0] - Yc[0];
    c1 += d1 * d1;
    Xc += X->inc;
    Yc += Y->inc;
  }    
  if (i < N) {
    d0 = Xc[0] - Yc[0];
    c0 += d0 * d0;
  }
  return c0 + c1;
}

void dvec_swap(mvec_t *X,  mvec_t *Y, int N)
{
  register int i;
  register double tmp;
  register double *Xc, *Yc;

  Xc = X->md;
  Yc = Y->md;
  for (i = 0; i < N-1; i += 2) {
    tmp = Xc[0];
    Xc[0] = Yc[0];
    Yc[0] = tmp;
    Xc += X->inc;
    Yc += Y->inc;
    tmp = Xc[0];
    Xc[0] = Yc[0];
    Yc[0] = tmp;
    Xc += X->inc;
    Yc += Y->inc;
  }    
  if (i < N) {
    tmp = Xc[0];
    Xc[0] = Yc[0];
    Yc[0] = tmp;
  }
}

void dvec_invscal(mvec_t *X,  double alpha, int N)
{
  register int i;
  register double *Xc;

  Xc = X->md;
  for (i = 0; i < N-1; i += 2) {
    Xc[0] /= alpha;
    Xc += X->inc;
    Xc[0] /= alpha;
    Xc += X->inc;
  }    
  if (i < N) {
    Xc[0] /= alpha;
  }
}

void dvec_scal(mvec_t *X,  double alpha, int N)
{
  register int i;
  register double *Xc;

  Xc = X->md;
  for (i = 0; i < N-1; i += 2) {
    Xc[0] *= alpha;
    Xc += X->inc;
    Xc[0] *= alpha;
    Xc += X->inc;
  }    
  if (i != N) {
    Xc[0] *= alpha;
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
