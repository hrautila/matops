
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef _INNER_DDOT_TRANS_H
#define _INNER_DDOT_TRANS_H


static inline
void _inner_ddot_trans(double *Cr, const double *Ar, const double *Br,
                       double alpha, int nVP, int ldB)
{
  register int k, iB;
  register double f0, f1, f2, f3, cval;

  cval = 0.0;
  // unrolling of loops;
  for (k = 0; k < nVP-3; k += 4) {
    iB = 0;
    f0 = Ar[0] * Br[iB];
    cval += f0;
    iB += ldB;
    f1 = Ar[1] * Br[iB];
    cval += f1;
    iB += ldB;
    f2 = Ar[2] * Br[iB];
    cval += f2;
    iB += ldB;
    f3 = Ar[3] * Br[iB];
    cval += f3;
    Br += 4*ldB;
    Ar += 4;
  }
  if (k == nVP)
    goto update;

  if (k < nVP-1) {
    iB = 0;
    f0 = Ar[0] * Br[0];
    cval += f0;
    iB += ldB;
    f1 = Ar[1] * Br[iB];
    cval += f1;
    Br += 2*ldB;
    Ar += 2;
    k += 2;
  }
  if (k < nVP) {
    f0 = Ar[0] * Br[0];
    cval += f0;
    Br += ldB;
    Ar++;
    k++;
  }
 update:
  f0 = cval * alpha;
  Cr[0] += f0;
}


static inline void
_inner_axpy_trans_nr(double *b0, const double *Ar, const double *Br, double alpha, int nC, int ldB)
{
  register int i;
  register double f0;
  f0 = alpha * Br[0];
  for (i = 0; i < nC; i++) {
    b0[0] += f0 * Ar[0];
    b0 += ldB;
    Ar++;
  }
}

static inline void
_inner_axpy_trans(double *b0, const double *Ar, const double *Br, double alpha, int nC, int ldB)
{
  register int k, iB;
  register double f0;

  f0 = alpha * Br[0];
  for (k = 0; k < nC-3; k += 4) {
    iB = 0;
    b0[iB] += f0 * Ar[0];
    iB += ldB;
    b0[iB] += f0 * Ar[1];
    iB += ldB;
    b0[iB] += f0 * Ar[2];
    iB += ldB;
    b0[iB] += f0 * Ar[3];
    Ar += 4;
    b0 += (ldB << 2);
  }
  if (k == nC)
    return;
  if (k < nC-1) {
    iB = 0;
    b0[iB] += f0 * Ar[0];
    iB += ldB;
    b0[iB] += f0 * Ar[1];
    Ar += 2;
    b0 += (ldB << 1);
    k += 2;
  }
  if (k < nC) {
    b0[0] += f0 * Ar[0];
  }
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
