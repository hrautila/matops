
#ifndef _MULT_NOSIMD_H
#define _MULT_NOSIMD_H 1

// update single element of C;
static inline
void mult1x1x4(double *c, const double *a, const double *b, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3;
  y0 = 0.0;
  y1 = y0; y2 = y0; y3 = y0;
  for (k = 0; k < nR-3; k += 4) {
    y0 += a[k]*b[k];
    y1 += a[k+1]*b[k+1];
    y2 += a[k+2]*b[k+2];
    y3 += a[k+3]*b[k+3];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    y0 += a[k]*b[k];
    k++;
  case 2:
    y1 += a[k]*b[k];
    k++;
  case 1:
    y2 += a[k]*b[k];
    break;
  }
update:
  y0 += y1; y2 += y3; y0 += y2;
  c[0] += y0*alpha;
}


// update 1x2 block of C
static inline
void mult2x1x4(double *c0, double *c1,
               const double *a,
               const double *b0, const double *b1, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 =  y2 = y3 = y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR-3; k += 4) {
    y0 += a[k+0]*b0[k+0];
    y1 += a[k+0]*b1[k+0];
    y2 += a[k+1]*b0[k+1];
    y3 += a[k+1]*b1[k+1];
    y4 += a[k+2]*b0[k+2];
    y5 += a[k+2]*b1[k+2];
    y6 += a[k+3]*b0[k+3];
    y7 += a[k+3]*b1[k+3];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    y0 += a[k]*b0[k];
    y1 += a[k]*b1[k];
    k++;
  case 2:
    y2 += a[k]*b0[k];
    y3 += a[k]*b1[k];
    k++;
  case 1:
    y4 += a[k]*b0[k];
    y5 += a[k]*b1[k];
    k++;
  }
update:
  y0 += y2 + y4 + y6;
  y1 += y3 + y5 + y7;
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
}

// update 2x2 block of C
static inline
void mult2x2x2(double *c0, double *c1, 
               const double *a0, const double *a1,
               const double *b0, const double *b1,
               double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 = y2 = y3 = 0.0;
  y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR-1; k += 2) {
    y0 += a0[k+0]*b0[k+0];
    y1 += a0[k+0]*b1[k+0];
    y2 += a0[k+1]*b0[k+1];
    y3 += a0[k+1]*b1[k+1];
    y4 += a1[k+0]*b0[k+0];
    y5 += a1[k+0]*b1[k+0];
    y6 += a1[k+1]*b0[k+1];
    y7 += a1[k+1]*b1[k+1];
  }
  if (k == nR)
    goto update;

  y0 += a0[k+0]*b0[k+0];
  y1 += a0[k+0]*b1[k+0];
  y4 += a1[k+0]*b0[k+0];
  y5 += a1[k+0]*b1[k+0];
  k++;

update:
  c0[0] += (y0 + y2)*alpha;
  c1[0] += (y1 + y3)*alpha;
  c0[1] += (y4 + y6)*alpha;
  c1[1] += (y5 + y7)*alpha;
}



// update 1x4 block of C;
static inline
void mult4x1x1(double *c0, double *c1, double *c2, double *c3,
               const double *a,
               const double *b0, const double *b1,
               const double *b2, const double *b3, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += a[k]*b0[k];
    y1 += a[k]*b1[k];
    y2 += a[k]*b2[k];
    y3 += a[k]*b3[k];
  }
update:
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
  c2[0] += y2*alpha;
  c3[0] += y3*alpha;
}

// update 2x4 block of C;
static inline
void mult4x2x1(double *c0, double *c1, double *c2, double *c3,
               const double *a0, const double *a1,
               const double *b0, const double *b1,
               const double *b2, const double *b3, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 = y2 = y3 = 0.0;
  y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += a0[k]*b0[k];
    y1 += a0[k]*b1[k];
    y2 += a0[k]*b2[k];
    y3 += a0[k]*b3[k];
    y4 += a1[k]*b0[k];
    y5 += a1[k]*b1[k];
    y6 += a1[k]*b2[k];
    y7 += a1[k]*b3[k];
  }
update:
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
  c2[0] += y2*alpha;
  c3[0] += y3*alpha;
  c0[1] += y4*alpha;
  c1[1] += y5*alpha;
  c2[1] += y6*alpha;
  c3[1] += y7*alpha;
}



// update 4x1 block of C;
static inline
void dmult4x1x1(double *c0, 
                const double *a0, const double *a1, const double *a2, const double *a3,
                const double *b0, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += b0[k]*a0[k];
    y1 += b0[k]*a1[k];
    y2 += b0[k]*a2[k];
    y3 += b0[k]*a3[k];
  }
update:
  c0[0] += y0*alpha;
  c0[1] += y1*alpha;
  c0[2] += y2*alpha;
  c0[3] += y3*alpha;
}


// update 4x2 block of C;
static inline
void dmult4x2x1(double *c0, double *c1,
                const double *a0, const double *a1, const double *a2, const double *a3,
                const double *b0, const double *b1, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 = y2 = y3 = 0.0;
  y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += b0[k]*a0[k];
    y1 += b0[k]*a1[k];
    y2 += b0[k]*a2[k];
    y3 += b0[k]*a3[k];
    y4 += b1[k]*a0[k];
    y5 += b1[k]*a1[k];
    y6 += b1[k]*a2[k];
    y6 += b1[k]*a3[k];

  }
update:
  c0[0] += y0*alpha;
  c0[1] += y1*alpha;
  c0[2] += y2*alpha;
  c0[3] += y3*alpha;
  c1[0] += y4*alpha;
  c1[1] += y5*alpha;
  c1[2] += y6*alpha;
  c1[3] += y7*alpha;
}


// update 2x1 block of C;
static inline
void dmult2x1x2(double *c0, 
                const double *a0, const double *a1, 
                const double *b0, double alpha, int nR)
{
  register int k;
  register double y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR-1; k += 2) {
    y0 += b0[k+0]*a0[k+0];
    y1 += b0[k+0]*a1[k+0];
    y2 += b0[k+1]*a0[k+1];
    y3 += b0[k+1]*a0[k+1];
  }
  if (k == nR)
    goto update;

  y0 += b0[k]*a0[k];
  y1 += b0[k]*a1[k];

update:
  c0[0] += (y0 + y2)*alpha;
  c0[1] += (y1 + y3)*alpha;
}

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:

