
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "testing"
)

func TestCompile(t *testing.T) {
	t.Logf("Compile OK\n")
}


func TestViewUpdate(t *testing.T) {
    Adata2 := [][]float64{
        []float64{4.0, 2.0, 2.0},
        []float64{6.0, 4.0, 2.0},
        []float64{4.0, 6.0, 1.0},
    }

    A := matrix.FloatMatrixFromTable(Adata2, matrix.RowOrder)
    N := A.Rows()

    // simple LU decomposition without pivoting
    var A11, a10, a01, a00 matrix.FloatMatrix
    for k := 1; k < N; k++ {
        a00.SubMatrixOf(A, k-1, k-1, 1, 1)
        a01.SubMatrixOf(A, k-1, k,   1, A.Cols()-k)
        a10.SubMatrixOf(A, k,   k-1, A.Rows()-k, 1)
        A11.SubMatrixOf(A, k,   k)
        //t.Logf("A11: %v  a01: %v\n", A11, a01)
        a10.Scale(1.0/a00.Float())
        MVRankUpdate(&A11, &a10, &a01, -1.0)
    }

    Ld := TriLU(A.Copy())
    Ud := TriU(A)
    t.Logf("Ld:\n%v\nUd:\n%v\n", Ld, Ud)
    An := matrix.FloatZeros(N, N)
    Mult(An, Ld, Ud, 1.0, 1.0, NOTRANS)
    t.Logf("A == Ld*Ud: %v\n", An.AllClose(An))
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
