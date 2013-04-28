
package matops

import (
    "github.com/hrautila/matrix"
    "math"
)

// Compute the UT HouseHolder transformation. Adapted from libFLAME FLA_Househ2_UT()
func householderUT(chi1, x2, tau *matrix.FloatMatrix, flags Flags) {

    // chi1 is 1x1. x2 is nx1, tau is 1x1

    if ! (flags & LEFT != 0 || flags & RIGHT != 0) {
        return
    }

    // norm_x2 = ||x2||_2
    norm_x2 := Norm2(x2)

    if norm_x2 == 0.0 {
        chi1.SetAt(0, 0, -chi1.Float())
        tau.SetAt(0, 0, 0.5)
        return
    }
    // abs_chi1 = |chi1| = ||chi1||_2
    chi1_val := chi1.Float()
    abs_chi1 := Norm2(chi1) //math.Abs(chi1_val)

    // norm_x = || [chi1; x2].T ||_2
    norm_x := math.Sqrt(abs_chi1*abs_chi1 + norm_x2*norm_x2)
    
    // alpha: = - ||x||_2 * chi1 / |chi1| 
    //        = - sign(chi1) * ||x||_2
    ssign := 1.0    
    if chi1_val < 0.0 {
        ssign = -1.0
    }
    alpha := -ssign * norm_x

    chi1_alpha := chi1_val - alpha

    // x2 = x2 / (chi1 - alpha)
    InvScale(x2, chi1_alpha)

    absq_chi1 := math.Abs(chi1_val - alpha)
    //absq_chi1 *= absq_chi1
    chi1_alpha *= chi1_alpha
    tau_val := sqrtX2Y2(absq_chi1, norm_x2)/(2.0*chi1_alpha)

    tau.SetAt(0, 0, tau_val)
    chi1.SetAt(0, 0, alpha)
}

func zeroDims(a *matrix.FloatMatrix) bool {
    return a.Rows() == 0 || a.Cols() == 0
}

// Apply a single Householder transform H 
//
// flags&LEFT != 0:
//   |a1| =  H'*|a1|
//   |A2|       |A2|
//
//   a1 = a1 - inv(tau)*( a1 + u2'*A2)
//   A2 = A2  - u2*inv(tau)*( a1 + u2'*A2)
//
// flags&RIGHT != 0:
//   |a1; A2| =  |a1; A2|*H
//
//   a1 = a1 - inv(tau)*( a1 + A2*u2)
//   A2 = A2 - u2*inv(tau)*( a1 + A2*u2)
//
// Adapted from libFLAME.
func householderApplyUT(tau, u2, a1, A2 *matrix.FloatMatrix, flags Flags) {

    // MVMult is vector orientation agnostic
    
    // w1 = a1 + u2'*A2;
    w1 := a1.Copy()
    if flags & LEFT != 0 {
        MVMult(w1, A2, u2, 1.0, 1.0, TRANSA)
    } else {
        MVMult(w1, A2, u2, 1.0, 1.0, NOTRANS)
    }        
    // w1 = w1/tau
    InvScale(w1, tau.Float())

    // a1 = a1 - w1
    a1.Minus(w1)

    // A2 = A2 - u2*w1
    MVRankUpdate(A2, u2, w1, -1.0)
}

/* From LAPACK/dlapy2.f
 *
 *> sqrtX2Y2() returns sqrt(x**2+y**2), taking care not to cause unnecessary
 *> overflow.
 */
func sqrtX2Y2(x, y float64) float64 {
    xabs := math.Abs(x)
    yabs := math.Abs(y)
    w := xabs
    if yabs > w {
        w = yabs
    }
    z := xabs
    if yabs < z {
        z = yabs
    }
    if z == 0.0 {
        return w
    }
    return w * math.Sqrt(1.0 + (z/w)*(z/w))
}

/* From LAPACK/dlarfg.f
 *>
 *> DLARFG generates a real elementary reflector H of order n, such
 *> that
 *>
 *>       H * ( alpha ) = ( beta ),   H**T * H = I.
 *>           (   x   )   (   0  )
 *>
 *> where alpha and beta are scalars, and x is an (n-1)-element real
 *> vector. H is represented in the form
 *>
 *>       H = I - tau * ( 1 ) * ( 1 v**T ) ,
 *>                     ( v )
 *>
 *> where tau is a real scalar and v is a real (n-1)-element
 *> vector.
 *>
 *> If the elements of x are all zero, then tau = 0 and H is taken to be
 *> the unit matrix.
 *>
 *> Otherwise  1 <= tau <= 2.
 */
func computeHouseholder(a11, x, tau *matrix.FloatMatrix, flags Flags) {
    
    // norm_x2 = ||x||_2
    norm_x2 := Norm2(x)
    if norm_x2 == 0.0 {
        //a11.SetAt(0, 0, -a11.GetAt(0, 0))
        tau.SetAt(0, 0, 0.0)
        return
    }

    alpha := a11.GetAt(0, 0)
    sign := 1.0
    if math.Signbit(alpha) {
        sign = -1.0
    }
    // beta = -(alpha / |alpha|) * ||alpha x||
    //      = -sign(alpha) * sqrt(alpha**2, norm_x2**2)
    beta := -sign*sqrtX2Y2(alpha, norm_x2)

    // x = x /(a11 - beta)
    InvScale(x, alpha-beta)

    tau.SetAt(0, 0, (beta-alpha)/beta)
    a11.SetAt(0, 0, beta)
}

/* From LAPACK/dlarf.f
 *
 *> Applies a real elementary reflector H to a real m by n matrix A,
 *> from either the left or the right. H is represented in the form
 *>
 *       H = I - tau * ( 1 ) * ( 1 v.T )
 *                     ( v )
 *
 *> where tau is a real scalar and v is a real vector.
 *>
 *> If tau = 0, then H is taken to be the unit matrix.
 *
 * A is /a1\   a1 := a1 - w1
 *      \A2/   A2 := A2 - v*w1
 *             w1 := tau*(a1 + A2.T*v) if side == LEFT
 *                := tau*(a1 + A2*v)   if side == RIGHT
 *
 */
func applyHouseholder(tau, v, a1, A2 *matrix.FloatMatrix, flags Flags) {

    tval := tau.GetAt(0, 0)
    if tval == 0.0 {
        return
    }
    w1 := a1.Copy()
    if flags & LEFT != 0 {
        // w1 = a1 + A2.T*v
        MVMult(w1, A2, v, 1.0, 1.0, TRANSA)
    } else {
        // w1 = a1 + A2*v
        MVMult(w1, A2, v, 1.0, 1.0, NOTRANS)
    }

    // w1 = tau*w1
    Scale(w1, tval)

    // a1 = a1 - w1
    a1.Minus(w1)

    // A2 = A2 - v*w1
    MVRankUpdate(A2, v, w1, -1.0)
}

func applyHouseholder0(tau, v, A2 *matrix.FloatMatrix, flags Flags) {

    tval := tau.GetAt(0, 0)
    if tval == 0.0 {
        return
    }
    w1 := matrix.FloatZeros(A2.Cols(), 1)
    if flags & LEFT != 0 {
        // w1 = A2.T*v
        MVMult(w1, A2, v, 1.0, 1.0, TRANSA)
    } else {
        // w1 = A2*v
        MVMult(w1, A2, v, 1.0, 1.0, NOTRANS)
    }

    // A2 = A2 - tau*v*w1
    MVRankUpdate(A2, v, w1, -tval)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

