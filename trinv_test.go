
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
	"github.com/hrautila/matrix"
	"testing"
)


func TestInvUpper(t *testing.T) {
	N := 30
	nb := 8
    A := matrix.FloatUniformSymmetric(N, matrix.Upper)
    I := matrix.FloatDiagonal(N, 1.0)
    I0 := matrix.FloatZeros(N, N)

    A0 := A.Copy()
    // A0 = A.-1
    InverseTrm(A0, UPPER, 0)
    Mult(I0, A, A0, 1.0, 0.0, NOTRANS)
    t.Logf("unblk: A*inv(A) == I: %v\n", I0.AllClose(I))

    A0 = A.Copy()
    // A0 = A.-1
    InverseTrm(A0, UPPER, nb)
    Mult(I0, A, A0, 1.0, 0.0, NOTRANS)
    t.Logf("blk  : A*inv(A) == I: %v\n", I0.AllClose(I))
}

func TestInvLower(t *testing.T) {
	N := 30
	nb := 8
    A := matrix.FloatUniformSymmetric(N, matrix.Lower)
    I := matrix.FloatDiagonal(N, 1.0)
    I0 := matrix.FloatZeros(N, N)

    A0 := A.Copy()
    // A0 = A.-1
    InverseTrm(A0, LOWER, 0)
    Mult(I0, A, A0, 1.0, 0.0, NOTRANS)
    t.Logf("unblk: A*inv(A) == I: %v\n", I0.AllClose(I))

    A0 = A.Copy()
    // A0 = A.-1
    InverseTrm(A0, LOWER, nb)
    Mult(I0, A, A0, 1.0, 0.0, NOTRANS)
    t.Logf("blk  : A*inv(A) == I: %v\n", I0.AllClose(I))
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
