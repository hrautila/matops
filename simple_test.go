
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "github.com/hrautila/linalg"
    "github.com/hrautila/linalg/blas"
    "testing"
)

func TestCompile(t *testing.T) {
	t.Logf("Compile OK\n")
}

func TestMVTransA(t *testing.T) {
    bM := 1000
    bN := 1000
    /*
    Adata := [][]float64{
     []float64{1.0, 1.0, 1.0, 1.0, 1.0},
     []float64{2.0, 2.0, 2.0, 2.0, 2.0},
     []float64{3.0, 3.0, 3.0, 3.0, 3.0},
     []float64{4.0, 4.0, 4.0, 4.0, 4.0},
     []float64{5.0, 5.0, 5.0, 5.0, 5.0}}
    A := matrix.FloatMatrixFromTable(Adata)
     */
    A := matrix.FloatNormal(bN, bM)
    X := matrix.FloatNormal(bN, 1)
    //X := matrix.FloatWithValue(bN, 1, 1.0)
    //A := matrix.FloatWithValue(bM, bN, 2.0)
    //X := matrix.FloatVector([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    blas.GemvFloat(A, X, Y0, 1.0, 1.0, linalg.OptTrans)

    MVMultTransA(Y1, A, X, 1.0, 1.0)
    t.Logf("Y0 == Y1: %v\n", Y0.AllClose(Y1))
    if bM <= 10 {
        t.Logf("blas: Y0 = A.T*X\n%v\n", Y0)
        t.Logf("Y1: Y1 = A*X\n%v\n", Y1)
    }
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
