
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


// Kahan summation for DOT product:
//    http://en.wikipedia.org/wiki/Kahan_summation_algorithm
// Not usually used in BLAS because of perfomance considerations.

// this is __not__ kahan summation 
double ddot_vec(const double *X, const double *Y, int incX, int incY, int N)
{
  register int i;
  register double c0, c1;
  c0 = 0.0; c1 = 0.0;
  for (i = 0; i < N-1; i += 2) {
    c0 += X[0] * Y[0];
    X += incX;
    Y += incY;
    c1 += X[0] * Y[0];
    X += incX;
    Y += incY;
  }    
  if (i == N) {
    return c0 + c1;
  }
  c0 += X[0] * Y[0];
  return c0 + c1;
}

// Scale a vector of N elements with incX interval.
void dscale_vec(double *X, int incX, double f0, int N)
{
  register int i;
  if (f0 == 1.0) {
    return;
  }
  if (f0 == 0.0) {
    for (i = 0; i < N; i++) {
      X[0] = 0.0;
      X += incX;
    }
    return;
  }
  for (i = 0; i < N; i++) {
    X[0] *= f0;
    X += incX;
  }
}

// Scale a tile of M rows by N columns with leading index ldX.
void dscale_tile(double *X, int ldX, double f0, int M, int N)
{
  register double *Xr, *Xc;
  register int i, j;
  if (f0 == 1.0) {
    return;
  }
  Xc = X;
  // set to zero
  if (f0 == 0.0) {
    for (j = 0; j < N; j++) {
      Xr = Xc;
      for (i = 0; i < M; i++) {
        Xr[0] = 0.0;
        Xr++;
      }
      Xc += ldX;
    }
    return;
  }

  // scale here
  for (j = 0; j < N; j++) {
    Xr = Xc;
    for (i = 0; i < M-3; i += 4) {
      Xr[0] *= f0;
      Xr[1] *= f0;
      Xr[2] *= f0;
      Xr[3] *= f0;
      Xr += 4;
    }
    if (i == M)
      goto increment;
    if (i < M-1) {
      Xr[0] *= f0;
      Xr[1] *= f0;
      Xr += 2;
      i += 2;
    }
    if (i < M) {
      Xr[0] *= f0;
      i++;
    }
  increment:
    Xc += ldX;
  }
  return;

}


// Local Variables:
// indent-tabs-mode: nil
// End:

