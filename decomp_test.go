
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "testing"
)

func _TestLU2x2NoPiv(t *testing.T) {
    Adata2 := [][]float64{
        []float64{4.0, 3.0},
        []float64{6.0, 3.0}}

    A := matrix.FloatMatrixFromTable(Adata2, matrix.RowOrder)
    DecomposeLUnoPiv(A, 0)
    t.Logf("A\n%v\n", A)
    Ld := TriLU(A.Copy())
    Ud := TriU(A)
    t.Logf("L*U\n%v\n", matrix.Times(Ld, Ud))
}

func _TestLU3x3NoPiv(t *testing.T) {
    Adata2 := [][]float64{
        []float64{4.0, 2.0, 2.0},
        []float64{6.0, 4.0, 2.0},
        []float64{4.0, 6.0, 1.0},
    }

    A := matrix.FloatMatrixFromTable(Adata2, matrix.RowOrder)
    A0 := A.Copy()
    DecomposeLUnoPiv(A, 0)
    t.Logf("A\n%v\n", A)
    Ld := TriLU(A.Copy())
    Ud := TriU(A.Copy())
    t.Logf("A == L*U: %v\n", A0.AllClose(matrix.Times(Ld, Ud)))
}

func _TestUnblkLUnoPiv(t *testing.T) {
    N := 6
    L := matrix.FloatUniformSymmetric(N, matrix.Lower)
    U := matrix.FloatUniformSymmetric(N, matrix.Upper)
    // Set L diagonal to 1.0
    L.Diag().SetIndexes(1.0)

    A := matrix.Times(L, U)
    t.Logf("A\n%v\n", A)
    R, _ := DecomposeLUnoPiv(A.Copy(), 0)
    Ld := TriLU(R.Copy())
    Ud := TriU(R)
    t.Logf("A == L*U: %v\n", A.AllClose(matrix.Times(Ld, Ud)))
}

func TestBlkLUnoPiv(t *testing.T) {
    N := 10
    nb := 4
    L := matrix.FloatUniformSymmetric(N, matrix.Lower)
    U := matrix.FloatUniformSymmetric(N, matrix.Upper)
    // Set L diagonal to 1.0
    L.Diag().SetIndexes(1.0)

    A := matrix.Times(L, U)
    t.Logf("A\n%v\n", A)
    R, _ := DecomposeLUnoPiv(A.Copy(), nb)
    Ld := TriLU(R.Copy())
    Ud := TriU(R)
    t.Logf("A == L*U: %v\n", A.AllClose(matrix.Times(Ld, Ud)))
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
