
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
    //"math/rand"
    //"math"
    //"time"
)


func _TestMultSmall(t *testing.T) {
    bM := 6
    bN := 6
    bP := 6
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatWithValue(bM, bN, 1.0)
    C1 := C0.Copy()

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 1.0)
    t.Logf("blas: C=D*E\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 1.0, NOTRANS, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}


func TestMultBig(t *testing.T) {
    bM := 100*M + 3
    bN := 100*N + 3
    bP := 100*P + 3
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 1.0)
    //t.Logf("blas: C=D*E\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 1.0, NOTRANS, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    res := C0.AllClose(C1)
    t.Logf("C0 == C1: %v\n", res)
}


func _TestMultTransASmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatWithValue(bM, bN, 0.0)
    C1 := C0.Copy()
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()
    blas.GemmFloat(Dt, E, C0, 1.0, 1.0, linalg.OptTransA)
    t.Logf("blas: C=D*E\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSA, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}


func _TestMultTransABig(t *testing.T) {
    bM := 100*M + 3
    bN := 100*N + 3
    bP := 100*P + 3
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(Dt, E, C0, 1.0, 1.0, linalg.OptTransA)

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSA, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestMultTransBSmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatWithValue(bP, bN, 1.0)
    C1 := C0.Copy()
    Et := E.Transpose()

    Dr := D.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, Et, C0, 1.0, 1.0, linalg.OptTransB)
    t.Logf("blas: C=D*E.T\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSB, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E.T\n%v\n", C1)
}


func _TestMultTransBBig(t *testing.T) {
    bM := 100*M + 3
    bN := 100*N + 3
    bP := 100*P + 3
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    C0 := matrix.FloatZeros(bM, bN)
    C1 := matrix.FloatZeros(bM, bN)
    Et := E.Transpose()

    Dr := D.FloatArray()
    Er := Et.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, Et, C0, 1.0, 1.0, linalg.OptTransB)

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSB, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestMultTransABSmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
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
    t.Logf("blas: C=D.T*E.T\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSA|TRANSB, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D.T*E.T\n%v\n", C1)
}


func _TestMultTransABBig(t *testing.T) {
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

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSA|TRANSB, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestMultMVSmall(t *testing.T) {
    bM := 5
    bN := 5
    A := matrix.FloatNormal(bM, bN)
    X := matrix.FloatVector([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Y1r := Y1.FloatArray()

    blas.GemvFloat(A, X, Y0, 1.0, 1.0)
    t.Logf("blas: Y=A*X\n%v\n", Y0)

    DMultMV(Y1r, Ar, Xr, 1.0, 1.0, NOTRANS, 1, A.LeadingIndex(), 1, 0,  bN, 0,  bM, 4, 4)
    t.Logf("Y0 == Y1: %v\n", Y0.AllClose(Y1))
    t.Logf("Y1: Y1 = A*X\n%v\n", Y1)
}

func _TestMultMV(t *testing.T) {
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

    DMultMV(Y1r, Ar, Xr, 1.0, 1.0, NOTRANS, 1, A.LeadingIndex(), 1, 0,  bN, 0,  bM, 32, 32)
    t.Logf("Y0 == Y1: %v\n", Y0.AllClose(Y1))
    if ! Y0.AllClose(Y1) {
        y0 := Y0.SubMatrix(0, 0, 2, 1)
        y1 := Y1.SubMatrix(0, 0, 2, 1)
        t.Logf("y0=\n%v\n", y0)
        t.Logf("y1=\n%v\n", y1)
    }
}

func _TestMultMVTransASmall(t *testing.T) {
    bM := 10
    bN := 10
    A := matrix.FloatNormal(bN, bM)
    X := matrix.FloatWithValue(bN, 1, 1.0)
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Y1r := Y1.FloatArray()

    blas.GemvFloat(A, X, Y0, 1.0, 1.0, linalg.OptTrans)
    t.Logf("blas: Y=A.T*X\n%v\n", Y0)

    DMultMV(Y1r, Ar, Xr, 1.0, 1.0, TRANSA, 1, A.LeadingIndex(), 1, 0,  bN, 0,  bM, 4, 4)
    t.Logf("Y0 == Y1: %v\n", Y0.AllClose(Y1))
    t.Logf("Y1: Y1 = A*X\n%v\n", Y1)
}

func _TestMultMVTransA(t *testing.T) {
    bM := 1000*M
    bN := 1000*N
    A := matrix.FloatNormal(bN, bM)
    X := matrix.FloatWithValue(bN, 1, 1.0)
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Y1r := Y1.FloatArray()

    blas.GemvFloat(A, X, Y0, 1.0, 1.0, linalg.OptTrans)

    DMultMV(Y1r, Ar, Xr, 1.0, 1.0, TRANSA, 1, A.LeadingIndex(), 1, 0,  bN, 0,  bM, 4, 4)
    ok := Y0.AllClose(Y1)
    t.Logf("Y0 == Y1: %v\n", ok)
    if ! ok {
        y1 := Y1.SubMatrix(0, 0, 5, 1)
        t.Logf("Y1[0:5]:\n%v\n", y1)
        y0 := Y0.SubMatrix(0, 0, 5, 1)
        t.Logf("Y0[0:5]:\n%v\n", y0)
    }
}


func TestMultSymmSmall(t *testing.T) {
    //bM := 5
    bN := 7
    bP := 7
    Adata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0},
        []float64{0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0}}

    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    B := matrix.FloatWithValue(bN, bP, 2.0)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptUpper, linalg.OptRight)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, UPPER|RIGHT, bN, A.LeadingIndex(), bN, bN, 0,  bP, 0,  bN, 2, 2, 2)
    ok := C0.AllClose(C1)
    t.Logf("C0 == C1: %v\n", ok)
    if ! ok {
        t.Logf("A=\n%v\n", A)
        t.Logf("blas: C=A*B\n%v\n", C0)
        t.Logf("C1: C1 = A*X\n%v\n", C1)
    }
}

func TestMultSymmLowerSmall(t *testing.T) {
    //bM := 5
    bN := 7
    bP := 7
    Adata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}

    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    B := matrix.FloatNormal(bN, bP)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptLower, linalg.OptRight)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, LOWER|RIGHT, bN, A.LeadingIndex(), bN,
        bN, 0,  bP, 0,  bN, 2, 2, 2)
    ok := C0.AllClose(C1)
    t.Logf("C0 == C1: %v\n", ok)
    if ! ok {
        t.Logf("A=\n%v\n", A)
        t.Logf("blas: C=A*B\n%v\n", C0)
        t.Logf("C1: C1 = A*X\n%v\n", C1)
    }
}

func TestMultSymmUpper(t *testing.T) {
    //bM := 5
    bN := 100*N + 3
    bP := 100*P + 3
    A := matrix.FloatNormalSymmetric(bN, matrix.Upper)
    B := matrix.FloatNormal(bN, bP)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptUpper)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, UPPER|LEFT, bN, A.LeadingIndex(), bN,
        bN, 0,  bP, 0,  bN, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func TestMultSymmLower(t *testing.T) {
    //bM := 5
    bN := 100*N
    bP := 100*P
    A := matrix.FloatNormalSymmetric(bN, matrix.Lower)
    B := matrix.FloatNormal(bN, bP)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptLower)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, LOWER|LEFT, bN, A.LeadingIndex(), bN,
        bN, 0,  bP, 0,  bN, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}




// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
