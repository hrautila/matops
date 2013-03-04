
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

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
