
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "github.com/hrautila/linalg/lapack"
    "testing"
)

func _TestLU2x2NoPiv(t *testing.T) {
    Adata2 := [][]float64{
        []float64{4.0, 3.0},
        []float64{6.0, 3.0}}

    A := matrix.FloatMatrixFromTable(Adata2, matrix.RowOrder)
    DecomposeBlockSize(0)
    DecomposeLUnoPiv(A)
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
    DecomposeBlockSize(0)
    DecomposeLUnoPiv(A)
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
    DecomposeBlockSize(0)
    R, _ := DecomposeLUnoPiv(A.Copy())
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
    DecomposeBlockSize(nb)
    R, _ := DecomposeLUnoPiv(A.Copy())
    Ld := TriLU(R.Copy())
    Ud := TriU(R)
    t.Logf("A == L*U: %v\n", A.AllClose(matrix.Times(Ld, Ud)))
}

func TestLU3x3Piv(t *testing.T) {
    Adata2 := [][]float64{
        []float64{3.0, 2.0, 2.0},
        []float64{6.0, 4.0, 1.0},
        []float64{4.0, 6.0, 3.0},
    }
    A := matrix.FloatMatrixFromTable(Adata2, matrix.RowOrder)
    piv := make([]int, A.Rows())
    piv0 := make([]int32, A.Rows())
    A0 := A.Copy()
    t.Logf("start A\n%v\n", A)
    DecomposeBlockSize(0)
    DecomposeLU(A, piv)
    Ld := TriLU(A.Copy())
    Ud := TriU(A.Copy())
    t.Logf("A\n%v\n", A)
    t.Logf("Ld:\n%v\n", Ld)
    t.Logf("Ud:\n%v\n", Ud)
    t.Logf("piv: %v\n", piv)
    t.Logf("result:\n%v\n", matrix.Times(Ld, Ud))
    //t.Logf("A == L*U: %v\n", A0.AllClose(matrix.Times(Ld, Ud)))
    lapack.Getrf(A0, piv0)
    t.Logf("lapack result: piv0 %v\n%v\n", piv0, A0)
    t.Logf("A == A0: %v\n", A0.AllClose(A))
}

func TestLU3x4Piv(t *testing.T) {
    Adata2 := [][]float64{
        []float64{3.0, 2.0, 2.0, 1.0},
        []float64{6.0, 4.0, 1.0, 2.0},
        []float64{4.0, 6.0, 3.0, 3.0},
    }
    A := matrix.FloatMatrixFromTable(Adata2, matrix.RowOrder)
    piv := make([]int, A.Rows())
    piv0 := make([]int32, A.Rows())
    A0 := A.Copy()
    t.Logf("start A\n%v\n", A)
    DecomposeBlockSize(0)
    DecomposeLU(A, piv)
    t.Logf("piv: %v\n", piv)
    lapack.Getrf(A0, piv0)
    t.Logf("lapack result: piv0 %v\n%v\n", piv0, A0)
    t.Logf("A == A0: %v\n", A0.AllClose(A))
}

func TestBlkLUPiv(t *testing.T) {
    N := 10
    nb := 4
    L := matrix.FloatUniformSymmetric(N, matrix.Lower)
    U := matrix.FloatUniformSymmetric(N, matrix.Upper)
    // Set L diagonal to 1.0
    L.Diag().SetIndexes(1.0)

    A := matrix.Times(L, U)
    A0 := A.Copy()
    piv := make([]int, N, N)
    t.Logf("A\n%v\n", A)
    DecomposeBlockSize(nb)
    R, _ := DecomposeLU(A.Copy(), piv)
    t.Logf("piv: %v\n", piv)
    t.Logf("R\n%v\n", R)

    piv0 := make([]int32, N, N)
    lapack.Getrf(A0, piv0)
    t.Logf("lapack result: piv0 %v\n%v\n", piv0, A0)
    t.Logf("R == A0: %v\n", A0.AllClose(R))

    //Ld := TriLU(R.Copy())
    //Ud := TriU(R)
    //t.Logf("A == L*U: %v\n", A.AllClose(matrix.Times(Ld, Ud)))
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
