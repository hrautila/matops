
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


package calgo

import (
    "github.com/hrautila/matrix"
    "github.com/hrautila/linalg/blas"
    "github.com/hrautila/linalg"
    "testing"
    "math/rand"
    "time"
)

const M = 8
const N = 8
const P = 8

var A, At, B, Bt, C *matrix.FloatMatrix


func TestMakeData(t *testing.T) {
    rand.Seed(time.Now().UnixNano())
    A = matrix.FloatWithValue(M, P, 2.0)
    At = A.Transpose()
    B = matrix.FloatWithValue(P, N, 1.0)
    Bt = B.Transpose()
    C = matrix.FloatZeros(M, N)
}

func _TestUnAlignedSmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 1.0)
    t.Logf("blas: C=D*E\n%v\n", C0)

    MultUnAligned(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}

func _TestAlignedSmall(t *testing.T) {
    bM := 6
    bN := 6
    bP := 6
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 1.0)
    t.Logf("blas: C=D*E\n%v\n", C0)

    MultAligned(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}

func _TestAligned(t *testing.T) {
    bM := 100*M
    bN := 100*N
    bP := 100*P
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 1.0)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    MultAligned(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestUnAligned(t *testing.T) {
    bM := 100*M + 1
    bN := 100*N + 1
    bP := 100*P + 1
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 1.0)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    MultUnAligned(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestUnAlignedSmallTransA(t *testing.T) {
    bM := 5
    bN := 5
    bP := 5
    //Ddata := [][]float64{
    //[]float64{1.0, 1.0, 1.0, 1.0, 1.0},
    //[]float64{2.0, 2.0, 2.0, 2.0, 2.0},
    //[]float64{3.0, 3.0, 3.0, 3.0, 3.0},
    //[]float64{4.0, 4.0, 4.0, 4.0, 4.0},
    //[]float64{5.0, 5.0, 5.0, 5.0, 5.0}}

    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 2.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()
    t.Logf("Dt:\n%v\n", Dt)
    t.Logf("E:\n%v\n", E)
    blas.GemmFloat(Dt, E, C0, 1.0, 1.0, linalg.OptTransA)
    t.Logf("blas: C=D*E\n%v\n", C0)

    MultUnAlignedTransA(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}

func _TestAlignedSmallTransA(t *testing.T) {
    bM := 6
    bN := 6
    bP := 6
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, E, C0, 1.0, 1.0, linalg.OptTransA)
    t.Logf("blas: C=D*E\n%v\n", C0)

    MultAlignedTransA(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}

func _TestAlignedTransA(t *testing.T) {
    bM := 100*M
    bN := 100*N
    bP := 100*P
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, E, C0, 1.0, 1.0, linalg.OptTransA)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    MultAlignedTransA(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestUnAlignedTransA(t *testing.T) {
    bM := 100*M + 1
    bN := 100*N + 1
    bP := 100*P + 1
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, E, C0, 1.0, 1.0, linalg.OptTransA)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    MultUnAlignedTransA(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestUnAlignedSmallTransB(t *testing.T) {
    bM := 5
    bN := 5
    bP := 5
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Et := E.Transpose()

    Dr := D.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, Et, C0, 1.0, 1.0, linalg.OptTransA)
    t.Logf("blas: C=D*E.T\n%v\n", C0)

    MultUnAlignedTransB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E.T\n%v\n", C1)
}

func _TestAlignedSmallTransB(t *testing.T) {
    bM := 6
    bN := 6
    bP := 6
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Et := E.Transpose()

    Dr := D.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, Et, C0, 1.0, 1.0, linalg.OptTransA)
    t.Logf("blas: C=D*E.T\n%v\n", C0)

    MultAlignedTransB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E.T\n%v\n", C1)
}

func _TestAlignedTransB(t *testing.T) {
    bM := 100*M
    bN := 100*N
    bP := 100*P
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Et := E.Transpose()

    Dr := D.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, Et, C0, 1.0, 1.0, linalg.OptTransB)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    MultAlignedTransB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestUnAlignedTransB(t *testing.T) {
    bM := 100*M + 1
    bN := 100*N + 1
    bP := 100*P + 1
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Et := E.Transpose()

    Dr := D.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, Et, C0, 1.0, 1.0, linalg.OptTransB)
    //t.Logf("blas: C=D*E.T\n%v\n", C0)

    MultUnAlignedTransB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}


func _TestUnAlignedSmallTransAB(t *testing.T) {
    bM := 5
    bN := 5
    bP := 5
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()
    Et := E.Transpose()

    Dr := Dt.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, Et, C0, 1.0, 1.0, linalg.OptTransA, linalg.OptTransB)
    t.Logf("blas: C=D.T*E.T\n%v\n", C0)

    MultUnAlignedTransAB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D.T*E.T\n%v\n", C1)
}

func _TestAlignedSmallTransAB(t *testing.T) {
    bM := 6
    bN := 6
    bP := 6
    /*
    Ddata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
        []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
        []float64{6.0, 6.0, 6.0, 6.0, 6.0, 6.0}}
    bM := 4
    bN := 4
    bP := 4
    Ddata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0},
        []float64{2.0, 2.0, 2.0, 2.0},
        []float64{3.0, 3.0, 3.0, 3.0},
        []float64{4.0, 4.0, 4.0, 4.0}}
     */

    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    //D := matrix.FloatMatrixFromTable(Ddata)
    //E := D.Copy().Scale(2.0)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()
    Et := E.Transpose()

    Dr := Dt.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()
    //t.Logf("D.T:\n%v\nE.T:\n%v\n", Dt, Et)

    blas.GemmFloat(Dt, Et, C0, 1.0, 1.0, linalg.OptTransA, linalg.OptTransB)
    t.Logf("blas: C=D*E.T\n%v\n", C0)

    MultAlignedTransAB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E.T\n%v\n", C1)
}

func _TestAlignedTransAB(t *testing.T) {
    bM := 100*M
    bN := 100*N
    bP := 100*P
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()
    Et := E.Transpose()

    Dr := Dt.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, Et, C0, 1.0, 1.0, linalg.OptTransA, linalg.OptTransB)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    MultAlignedTransAB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestUnAlignedTransAB(t *testing.T) {
    bM := 100*M + 1
    bN := 100*N + 1
    bP := 100*P + 1
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()
    Et := E.Transpose()

    Dr := Dt.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, Et, C0, 1.0, 1.0, linalg.OptTransA, linalg.OptTransB)
    //t.Logf("blas: C=D*E.T\n%v\n", C0)

    MultUnAlignedTransAB(C1r, Dr, Er, 1.0, 1.0, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestCopyTrans(t *testing.T) {
    A := matrix.FloatNormal(4, 5);
    C := matrix.FloatZeros(6, 4);
    copy_trans(C.FloatArray(), A.FloatArray(), C.LeadingIndex(),
        A.LeadingIndex(), A.Rows(), A.Cols())

    t.Logf("A:\n%v\nC:\n%v\n", A, C);
}

func TestMatVecUnAlignedSmall(t *testing.T) {
    bM := 5
    bN := 5
    //A := matrix.FloatNormal(bM, bN)
    //X := matrix.FloatNormal(bN, 1)
    A := matrix.FloatWithValue(bM, bN, 2.0)
    X := matrix.FloatVector([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Y1r := Y1.FloatArray()

    blas.GemvFloat(A, X, Y0, 1.0, 1.0)
    t.Logf("blas: Y=A*X\n%v\n", Y0)

    MatVecUnAligned(Y1r, Ar, Xr, 1.0, 1.0, 1, A.LeadingIndex(), 1, 0,  bN, 0,  bM, 4, 4)
    t.Logf("Y0 == Y1: %v\n", Y0.AllClose(Y1))
    t.Logf("Y1: Y1 = A*X\n%v\n", Y1)
}

func TestMatVecUnAligned(t *testing.T) {
    bM := 100*M
    bN := 100*N
    A := matrix.FloatNormal(bM, bN)
    X := matrix.FloatNormal(bN, 1)
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Y1r := Y1.FloatArray()

    blas.GemvFloat(A, X, Y0, 1.0, 1.0)

    MatVecUnAligned(Y1r, Ar, Xr, 1.0, 1.0, 1, A.LeadingIndex(), 1, 0,  bN, 0,  bM, 32, 32)
    t.Logf("Y0 == Y1: %v\n", Y0.AllClose(Y1))
    if ! Y0.AllClose(Y1) {
        y0 := Y0.SubMatrix(0, 0, 2, 1)
        y1 := Y1.SubMatrix(0, 0, 2, 1)
        t.Logf("y0=\n%v\n", y0)
        t.Logf("y1=\n%v\n", y1)
    }
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
