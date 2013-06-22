
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
	"github.com/hrautila/matrix"
	"testing"
)


func TestLDLnoPiv(t *testing.T) {
	N := 42
    nb := 8

	A := matrix.FloatUniformSymmetric(N)
    B := matrix.FloatNormal(A.Rows(), 2)
    w := matrix.FloatWithValue(A.Rows(), 2, 1.0)

    // B0 = A*B
    B0 := B.Copy()

    nb = 2
    L, _ := DecomposeLDLnoPiv(A.Copy(), w, LOWER, nb)
    Mult(B0, A, B, 1.0, 0.0, NOTRANS)
    SolveLDLnoPiv(B0, L, LOWER)
    t.Logf("L*D*L.T: ||B - A*X||_1: %e\n", NormP(B0.Minus(B), NORM_ONE))

    U, _ := DecomposeLDLnoPiv(A.Copy(), w, UPPER, nb)
    Mult(B0, A, B, 1.0, 0.0, NOTRANS)
    SolveLDLnoPiv(B0, U, UPPER)
    t.Logf("U*D*U.T: ||B - A*X||_1: %e\n", NormP(B0.Minus(B), NORM_ONE))

}

func TestLDLlower(t *testing.T) {
	N := 8
    nb := 0

	A := matrix.FloatUniformSymmetric(N)
    B := matrix.FloatNormal(A.Rows(), 2)
    B0 := B.Copy()
    B1 := B.Copy()
    Mult(B0, A, B, 1.0, 0.0, NOTRANS)

    ipiv := make([]int, N, N)
    L, _ := DecomposeLDL(A.Copy(), nil, ipiv, LOWER, 0)

    ApplyRowPivots(B, ipiv, FORWARD)
    MultTrm(B, L, 1.0, LOWER|UNIT|TRANSA)
    MultDiag(B, L, LEFT)
    MultTrm(B, L, 1.0, LOWER|UNIT)
    ApplyRowPivots(B0, ipiv, FORWARD)
    t.Logf("unblk: L*D*L.T %d pivots: ||A*B - L*D*L.T*B||_1: %e\n",
        NumPivots(ipiv), NormP(B.Minus(B0), NORM_ONE))

    nb = 4
    w := matrix.FloatWithValue(A.Rows(), nb, 1.0)
    L, _ = DecomposeLDL(A.Copy(), w, ipiv, LOWER, nb)
    // B2 = A*B1 == A*B
    B2 := B1.Copy()
    Mult(B2, A, B1, 1.0, 0.0, NOTRANS)

    ApplyRowPivots(B1, ipiv, FORWARD)
    MultTrm(B1, L, 1.0, LOWER|UNIT|TRANSA)
    MultDiag(B1, L, LEFT)
    MultTrm(B1, L, 1.0, LOWER|UNIT)
    ApplyRowPivots(B2, ipiv, FORWARD)
    t.Logf("  blk: L*D*L.T %d pivots: ||A*B - L*D*L.T*B||_1: %e\n",
        NumPivots(ipiv), NormP(B2.Minus(B1), NORM_ONE))

}

func TestLDLupper(t *testing.T) {
	N := 8
    nb := 0

	A := matrix.FloatUniformSymmetric(N)
    B := matrix.FloatNormal(A.Rows(), 2)
    B0 := B.Copy()
    B1 := B.Copy()
    Mult(B0, A, B, 1.0, 0.0, NOTRANS)

    ipiv := make([]int, N, N)
    U, _ := DecomposeLDL(A.Copy(), nil, ipiv, UPPER, 0)

    ApplyRowPivots(B, ipiv, BACKWARD)
    MultTrm(B, U, 1.0, UPPER|UNIT|TRANSA)
    MultDiag(B, U, LEFT)
    MultTrm(B, U, 1.0, UPPER|UNIT)
    ApplyRowPivots(B0, ipiv, BACKWARD)
    t.Logf("unblk: U*D*U.T %d pivots: ||A*B - U*D*U.T*B||_1: %e\n",
        NumPivots(ipiv), NormP(B.Minus(B0), NORM_ONE))
    t.Logf("pivots: %v\n", ipiv)

    nb = 4
    w := matrix.FloatWithValue(A.Rows(), nb, 1.0)
    U, _ = DecomposeLDL(A.Copy(), w, ipiv, UPPER, nb)
    // B2 = A*B1 == A*B
    B2 := B1.Copy()
    Mult(B2, A, B1, 1.0, 0.0, NOTRANS)

    ApplyRowPivots(B1, ipiv, BACKWARD)
    MultTrm(B1, U, 1.0, UPPER|UNIT|TRANSA)
    MultDiag(B1, U, LEFT)
    MultTrm(B1, U, 1.0, UPPER|UNIT)
    ApplyRowPivots(B2, ipiv, BACKWARD)
    t.Logf("  blk: U*D*U.T %d pivots: ||A*B - U*D*U.T*B||_1: %e\n",
        NumPivots(ipiv), NormP(B2.Minus(B1), NORM_ONE))
    t.Logf("pivots: %v\n", ipiv)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
