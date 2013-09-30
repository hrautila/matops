
#ifndef _MATCPY_H
#define _MATCPY_H 1


#include <string.h>

static inline
void copy_plain_mcpy1(double *d, int ldD, const double *s, int ldS, int nR, int nC) {
  register int j;
  for (j = 0; j < nC; j ++) {
    memcpy(&d[(j+0)*ldD], &s[(j+0)*ldS], nR*sizeof(double));
  }
}

static inline
void copy_trans1x4(double *d, int ldD, const double *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC; j ++) {
    for (i = 0; i < nR-3; i += 4) {
      d[(j+0)+(i+0)*ldD] = s[(i+0)+(j+0)*ldS];
      d[(j+0)+(i+1)*ldD] = s[(i+1)+(j+0)*ldS];
      d[(j+0)+(i+2)*ldD] = s[(i+2)+(j+0)*ldS];
      d[(j+0)+(i+3)*ldD] = s[(i+3)+(j+0)*ldS];
    }
    if (i == nR)
      continue;
    switch (nR-i) {
    case 3:
      d[j+i*ldD] = s[i+j*ldS];
      i++;
    case 2:
      d[j+i*ldD] = s[i+j*ldS];
      i++;
    case 1:
      d[j+i*ldD] = s[i+j*ldS];
    }
  }
}

static inline
void copy_trans4x1(double *d, int ldD, const double *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC-3; j += 4) {
    for (i = 0; i < nR; i ++) {
      d[(j+0)+(i+0)*ldD] = s[i+(j+0)*ldS];
      d[(j+1)+(i+0)*ldD] = s[i+(j+1)*ldS];
      d[(j+2)+(i+0)*ldD] = s[i+(j+2)*ldS];
      d[(j+3)+(i+0)*ldD] = s[i+(j+3)*ldS];
    }
  }
  if (j == nC)
    return;
  copy_trans1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}



static inline
void __CPTRANS(double *d, int ldD, const double *s, int ldS, int nR, int nC) {
  copy_trans4x1(d, ldD, s, ldS, nR, nC);
}

static inline
void __CP(double *d, int ldD, const double *s, int ldS, int nR, int nC) {
  copy_plain_mcpy1(d, ldD, s, ldS, nR, nC);
}


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:
