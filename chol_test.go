
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "testing"
    //"math"
)


func TestUpperCHOL(t *testing.T) {
	N := 7
    K := 6
    nb := 4
    Z := matrix.FloatUniform(N, N)
	A := matrix.Times(Z, Z.Transpose())
	B := matrix.FloatUniform(N, K)
	X := B.Copy()

	// R = chol(A) = U.T*U
    R, _ := DecomposeCHOL(TriU(A.Copy()), UPPER, nb)

	// X = A.-1*B = U.-1*(U.-T*B)
	SolveCHOL(X, R, UPPER)

	// B = B - A*X
	Mult(B, A, X, -1.0, 1.0, NONE)

	// ||B - A*X||_2
	nrm := Norm2(matrix.FloatVector(B.FloatArray()))
	t.Logf("||B - A*X||_2: %e\n", nrm)
}


func TestLowerCHOL(t *testing.T) {
    N := 40
	K := 18
    nb := 8
    Z := matrix.FloatUniform(N, N)
	A := matrix.Times(Z, Z.Transpose())
	B := matrix.FloatUniform(N, K)
	X := B.Copy()

	// R = chol(A) = L*L.T
    R, _ := DecomposeCHOL(A.Copy(), LOWER, nb)

	// X = A.-1*B = L.-T*(L.-1*B)
	SolveCHOL(X, R, LOWER)

	// B = B - A*X
	Mult(B, A, X, -1.0, 1.0, NONE)

	// ||B - A*X||_2
	nrm := Norm2(matrix.FloatVector(B.FloatArray()))
	t.Logf("||B - A*X||_2: %e\n", nrm)
}



// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
