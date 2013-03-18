
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


func fail(t *testing.T, isok bool) {
    if ! isok {
        t.Errorf("** FAILURE HERE ***\n")
    }
}

func TestMakeData(t *testing.T) {
    blas.PanicOnError(true)
    rand.Seed(time.Now().UnixNano())
}

func _TestMultSmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
    /*
    Ddata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
        []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
        []float64{6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0},
        []float64{7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0}}
    D := matrix.FloatMatrixFromTable(Ddata, matrix.RowOrder)
     */
    //E := matrix.FloatMatrixFromTable(Ddata, matrix.RowOrder)
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 1.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    //C0 := matrix.FloatZeros(bM, bN)
    //C1 := matrix.FloatZeros(bM, bN)
    C0 := matrix.FloatWithValue(bM, bN, 1.0)
    C1 := C0.Copy()

    Dr := D.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()

    blas.GemmFloat(D, E, C0, 1.0, 2.0)
    t.Logf("blas: C=D*E\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 2.0, NOTRANS, bM, bM, bP, bP, 0,  bN, 0,  bM, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1=D*E\n%v\n", C1)
}


func _TestMultBig(t *testing.T) {
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
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}


func _TestMultTransASmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
    /*
    Ddata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
        []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
        []float64{6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0},
        []float64{7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0}}
    D := matrix.FloatMatrixFromTable(Ddata, matrix.RowOrder)
     */
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 2.0)
    //C0 := matrix.FloatZeros(bM, bN)
    //C1 := matrix.FloatZeros(bM, bN)
    C0 := matrix.FloatWithValue(bM, bN, 0.0)
    C1 := C0.Copy()
    Dt := D.Transpose()

    Dr := Dt.FloatArray()
    Er := E.FloatArray()
    C1r := C1.FloatArray()
    //t.Logf("Dt:\n%v\n", Dt)
    //t.Logf("E:\n%v\n", E)
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
    /*
    Ddata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
        []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
        []float64{6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0},
        []float64{7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0}}
    D := matrix.FloatMatrixFromTable(Ddata, matrix.RowOrder)
     */
    D := matrix.FloatNormal(bM, bP)
    E := matrix.FloatNormal(bP, bN)
    //D := matrix.FloatWithValue(bM, bP, 2.0)
    //E := matrix.FloatWithValue(bP, bN, 1.0)
    //C0 := matrix.FloatZeros(bM, bN)
    //C1 := matrix.FloatZeros(bM, bN)
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
    //t.Logf("blas: C=D*E\n%v\n", C0)

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSB, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func _TestMultTransABSmall(t *testing.T) {
    bM := 7
    bN := 7
    bP := 7
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
    //X := matrix.FloatNormal(bN, 1)
    //A := matrix.FloatWithValue(bM, bN, 2.0)
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
    //X := matrix.FloatNormal(bN, 1)
    X := matrix.FloatWithValue(bN, 1, 1.0)
    //A := matrix.FloatWithValue(bM, bN, 2.0)
    //X := matrix.FloatVector([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
    //At := A.Transpose()
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
    //X := matrix.FloatNormal(bN, 1)
    X := matrix.FloatWithValue(bN, 1, 1.0)
    Y1 := matrix.FloatZeros(bM, 1)
    Y0 := matrix.FloatZeros(bM, 1)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Y1r := Y1.FloatArray()

    blas.GemvFloat(A, X, Y0, 1.0, 1.0, linalg.OptTrans)
    //t.Logf("blas: Y=A.T*X\n%v\n", Y0)

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


func _TestMultSymmSmall(t *testing.T) {
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

    //A := matrix.FloatNormal(bN, bN)
    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    //B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    B := matrix.FloatWithValue(bN, bP, 2.0)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    t.Logf("A=\n%v\n", A)
    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptUpper, linalg.OptRight)
    t.Logf("blas: C=A*B\n%v\n", C0)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, UPPER|RIGHT, bN, A.LeadingIndex(), bN, bN, 0,  bP, 0,  bN, 2, 2, 2)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)
}

func _TestMultSymmLowerSmall(t *testing.T) {
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

    //A := matrix.FloatNormal(bN, bN)
    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    //B := matrix.FloatWithValue(bN, bP, 1.0)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    t.Logf("A=\n%v\n", A)
    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptLower, linalg.OptRight)
    t.Logf("blas: C=A*B\n%v\n", C0)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, LOWER|RIGHT, bN, A.LeadingIndex(), bN,
        bN, 0,  bP, 0,  bN, 2, 2, 2)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)
}

func _TestMultSymmUpper(t *testing.T) {
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

func _TestMultSymmLower(t *testing.T) {
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

func _TestRankSmall(t *testing.T) {
    bM := 5
    bN := 5
    //bP := 5
    Adata := [][]float64{
     []float64{1.0, 1.0, 1.0, 1.0, 1.0},
     []float64{2.0, 2.0, 2.0, 2.0, 2.0},
     []float64{3.0, 3.0, 3.0, 3.0, 3.0},
     []float64{4.0, 4.0, 4.0, 4.0, 4.0},
     []float64{5.0, 5.0, 5.0, 5.0, 5.0}}

    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    A0 := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    X := matrix.FloatVector([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
    Y := matrix.FloatWithValue(bN, 1, 2.0)

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Yr := Y.FloatArray()

    t.Logf("A=\n%v\n", A)
    blas.GerFloat(X, Y, A0, 1.0)
    t.Logf("blas ger:\n%v\n", A0)

    DRankMV(Ar, Xr, Yr, 1.0, A.LeadingIndex(), 1, 1, 0,  bN, 0,  bM, 4, 4)
    t.Logf("A0 == A1: %v\n", A0.AllClose(A))
    t.Logf("A1: \n%v\n", A)
}

func _TestRank(t *testing.T) {
    bM := M*100
    bN := N*100
    //bP := 5

    A := matrix.FloatWithValue(bM, bN, 1.0);
    A0 := matrix.FloatWithValue(bM, bN, 1.0);
    X := matrix.FloatNormal(bM, 1);
    Y := matrix.FloatNormal(bN, 1);

    Ar := A.FloatArray()
    Xr := X.FloatArray()
    Yr := Y.FloatArray()

    blas.GerFloat(X, Y, A0, 1.0)

    DRankMV(Ar, Xr, Yr, 1.0, A.LeadingIndex(), 1, 1, 0,  bN, 0,  bM, 4, 4)
    t.Logf("A0 == A1: %v\n", A0.AllClose(A))
}

func _TestMultSyrSmall(t *testing.T) {
    bN := 7
    //A := matrix.FloatNormal(bN, bN)
    //B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 1.0)
    X := matrix.FloatWithValue(bN, 1, 1.0)
    C0 := matrix.FloatZeros(bN, bN)
    C1 := matrix.FloatZeros(bN, bN)
    for i := 0; i < bN; i++ {
        X.Add(1.0+float64(i), i)
    }
    t.Logf("X=\n%v\n", X)

    Xr := X.FloatArray()
    C1r := C1.FloatArray()

    blas.SyrFloat(X, C0, 1.0, linalg.OptUpper)
    t.Logf("blas: C0\n%v\n", C0)

    DSymmRankMV(C1r, Xr, 1.0, UPPER, C1.LeadingIndex(), 1, 0,  bN, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)

    blas.SyrFloat(X, C0, 1.0, linalg.OptLower)
    t.Logf("blas: C0\n%v\n", C0)

    DSymmRankMV(C1r, Xr, 1.0, LOWER, C1.LeadingIndex(), 1, 0,  bN, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)
}

func _TestMultSyr2Small(t *testing.T) {
    bN := 7
    //A := matrix.FloatNormal(bN, bN)
    //B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 1.0)
    X := matrix.FloatWithValue(bN, 1, 1.0)
    Y := matrix.FloatWithValue(bN, 1, 1.0)
    C0 := matrix.FloatZeros(bN, bN)
    C1 := matrix.FloatZeros(bN, bN)
    for i := 0; i < bN; i++ {
        X.Add(1.0+float64(i), i)
        Y.Add(2.0+float64(i), i)
    }
    t.Logf("X=\n%v\nY=\n%v\n", X, Y)

    Xr := X.FloatArray()
    Yr := Y.FloatArray()
    C1r := C1.FloatArray()

    blas.Syr2Float(X, Y, C0, 1.0, linalg.OptUpper)
    t.Logf("blas: C0\n%v\n", C0)

    DSymmRank2MV(C1r, Xr, Yr, 1.0, UPPER, C1.LeadingIndex(), 1, 1, 0,  bN, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)

    blas.Syr2Float(X, Y, C0, 1.0, linalg.OptLower)
    t.Logf("blas: C0\n%v\n", C0)

    DSymmRank2MV(C1r, Xr, Yr, 1.0, LOWER, C1.LeadingIndex(), 1, 1, 0,  bN, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    
    t.Logf("C1: C1 = A*X\n%v\n", C1)
}


func solveMVTest(t *testing.T, A, X0 *matrix.FloatMatrix, flags Flags, bN, bNB int) {
    X1 := X0.Copy()

    uplo := linalg.OptUpper
    diag := linalg.OptNonUnit
    if flags & LOWER != 0 {
        uplo = linalg.OptLower
    }
    if flags & UNIT != 0 {
        diag = linalg.OptUnit
    }
        
    if bN < 8 {
        t.Logf("A=\n%v\n", A)
        t.Logf("X0=\n%v\n", X0)
    }
    blas.TrsvFloat(A, X0, uplo, diag)
    if bN < 8 {
        t.Logf("blas: X0\n%v\n", X0)
    }

    Ar := A.FloatArray()
    Xr := X1.FloatArray()
    if bN == bNB {
        DSolveUnblkMV(Xr, Ar, flags,  1, A.LeadingIndex(), bN)
    } else {
        DSolveBlkMV(Xr, Ar, flags,  1, A.LeadingIndex(), bN, bNB)
    }
    t.Logf("X1 == X0: %v\n", X1.AllClose(X0))
    if bN < 8 {
        t.Logf("X1:\n%v\n", X1)
    }
}


func _TestSolveSmall(t *testing.T) {
    Adata := [][]float64{
        []float64{1.0, 0.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0}}

    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    bN := A.Rows()
    At := A.Transpose()
    X0 := matrix.FloatWithValue(A.Rows(), 1, 1.0)
    X1 := X0.Copy()
    xsum := 0.0
    for i := 0; i < bN; i++ {
        xsum += float64(i)
        X0.Add(xsum, i)
        X1.Add(xsum, -(i+1))
    }
    X2 := X0.Copy()
    X3 := X0.Copy()

    t.Logf("-- SOLVE LOWER NON-UNIT ---\n")
    solveMVTest(t, A, X0, LOWER, A.Rows(), A.Rows())
    t.Logf("-- SOLVE LOWER UNIT ---\n")
    A.Diag().SetIndexes(1.0)
    solveMVTest(t, A, X1, LOWER|UNIT, A.Rows(), A.Rows())

    t.Logf("-- SOLVE UPPER NON-UNIT  ---\n")
    solveMVTest(t, At, X2, UPPER, At.Rows(), At.Rows())
    t.Logf("-- SOLVE UPPER UNIT ---\n")
    At.Diag().SetIndexes(1.0)
    solveMVTest(t, At, X3, UPPER|UNIT, At.Rows(), At.Rows())
}

func _TestSolveBlockedSmall(t *testing.T) {
    Adata := [][]float64{
        []float64{1.0, 0.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0}}

    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    X0 := matrix.FloatWithValue(A.Rows(), 1, 1.0)
    X1 := X0.Copy()
    X2 := X0.Copy()
    xsum := 0.0
    for i := 0; i < A.Rows(); i++ {
        xsum += float64(i)
        X0.Add(xsum, i)
        X2.Add(xsum, -(i+1))
    }
    t.Logf("-- SOLVE NON-UNIT ---\n")
    solveMVTest(t, A, X0, LOWER, A.Rows(), 4)
    solveMVTest(t, A.Transpose(), X2, UPPER, A.Rows(), 4)
    
    t.Logf("-- SOLVE UNIT ---\n")
    A.Diag().SetIndexes(1.0)
    solveMVTest(t, A, X1, UPPER|UNIT, A.Rows(), 4)
}

func TestSolveRandom(t *testing.T) {
    bN := 22
    A := matrix.FloatNormalSymmetric(bN, matrix.Lower)
    At := A.Transpose()
    X0 := matrix.FloatWithValue(A.Rows(), 1, 1.0)
    X1 := X0.Copy()
    X2 := X0.Copy()
    t.Logf("-- BLOCKED SOLVE LOWER NON-UNIT ---\n")
    solveMVTest(t, A, X0, LOWER, bN, 4)
    t.Logf("-- BLOCKED SOLVE LOWER UNIT ---\n")
    A.Diag().SetIndexes(1.0)
    solveMVTest(t, A, X1, LOWER|UNIT, bN, 4)
    t.Logf("-- BLOCKED SOLVE UPPER NON-UNIT ---\n")
    solveMVTest(t, At, X2, UPPER, bN, 4)
}


func tridiagSmall(t *testing.T, unit bool) {
    //bM := 5
    bN := 5
    Adata := [][]float64{
        []float64{1.0, 0.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0}}

    var uflag Flags = NULL
    if unit {
        uflag |= UNIT
    }
    diag := linalg.OptNonUnit
    if unit {
        diag = linalg.OptUnit
    }
    Al := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    //A := matrix.FloatNormal(bN, bN)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    //Z := matrix.FloatNormal(bN, 1);
    X0 := matrix.FloatWithValue(bN, 1, 2.0)
    X2 := matrix.FloatWithValue(bN, 1, 2.0)
    xsum := 0.0
    for i := 0; i < bN; i++ {
        xsum += float64(i) + 1.0
        //X0.Add(xsum, i)
        X2.Add(xsum, -(i+1))
    }
    //X0.Mul(Z)
    X1 := X0.Copy()
    //X2.Mul(Z)
    //X3 := X2.Copy()
    Au := Al.Transpose()

    t.Logf("X0=\n%v\n", X0)
    t.Logf("A(upper)=\n%v\n", Au)
    t.Logf("A(lower)=\n%v\n", Al)
    //t.Logf("Z=\n%v\n", Z)
    blas.TrmvFloat(Al, X0, linalg.OptUpper, diag)
    t.Logf("1. A(lower), blas(upper): X0 = Al*X0\n%v\n", X0)

    Ar := Al.FloatArray()
    Xr := X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, LOWER|uflag, 1, Al.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(lower), X1(fwd) = Al*X1:\n%v\n", X1)
    
    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)

    blas.TrmvFloat(Au, X0, linalg.OptUpper, diag)
    t.Logf("2. A(upper), blas(upper): X0 = Au*X0\n%v\n", X0)

    Ar = Au.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, UPPER|uflag, 1, Au.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(upper), X1(fwd) = Au*X1:\n%v\n", X1)

    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)
    blas.TrmvFloat(Al, X0, linalg.OptLower, diag)
    t.Logf("3. A(lower), blas(lower): X0 = Al*X0\n%v\n", X0)

    Ar = Al.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, LOWER|uflag, 1, Al.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(lower), X1(backwd) = Al*X1:\n%v\n", X1)

    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)
    blas.TrmvFloat(Au, X0, linalg.OptLower, diag)
    t.Logf("4. A(upper), blas(lower): X0 = Au*X0\n%v\n", X0)

    Ar = Au.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, LOWER|uflag, 1, Au.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(upper), X1(backwd) = Au*X1:\n%v\n", X1)

    t.Logf("-- TRANSPOSED --\n")
    Au_T := Al
    Al_T := Au
    t.Logf("A(upper).T=\n%v\n", Au_T)
    t.Logf("A(lower).T=\n%v\n\n", Al_T)

    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)

    blas.TrmvFloat(Au_T, X0, linalg.OptUpper, linalg.OptTrans, diag)
    t.Logf("5. A(upper).T, blas(upper,trans): X0 = Au.T*X0\n%v\n", X0)

    Ar = Au_T.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, UPPER|TRANSA|uflag, 1, Au_T.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(upper).T, X1(fwd,trans) = Au.T*X1:\n%v\n", X1)

    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)

    blas.TrmvFloat(Au_T, X0, linalg.OptLower, linalg.OptTrans, diag)
    t.Logf("6. A(upper).T, blas(lower,trans): X0 = Au.T*X0\n%v\n", X0)

    Ar = Au_T.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, LOWER|TRANSA|uflag, 1, Au_T.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(upper).T, X1(backwd,transA) = Au.T*X1:\n%v\n", X1)

    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)

    blas.TrmvFloat(Al_T, X0, linalg.OptLower, linalg.OptTrans, diag)
    t.Logf("7. A(lower).T, blas(lower,trans): X0 = Al.T*X0\n%v\n", X0)

    Ar = Al_T.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, LOWER|TRANSA|uflag, 1, Al.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(lower).T, X1(backwd,trans) = Al.T*X1:\n%v\n", X1)

    X0.SetIndexes(2.0)
    X1.SetIndexes(2.0)

    blas.TrmvFloat(Al_T, X0, linalg.OptUpper, linalg.OptTrans, diag)
    t.Logf("8. A(lower).T, blas(upper,trans): X0 = Al.T*X0\n%v\n", X0)

    Ar = Al_T.FloatArray()
    Xr = X1.FloatArray()
    DTrimvUnblkMV(Xr, Ar, UPPER|TRANSA|uflag, 1, Al_T.LeadingIndex(), bN)
    t.Logf("   X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("   A(lower).T, X1(fwd,trans) = Al.T*X1:\n%v\n", X1)

}

func _TestTridiagNonUnitSmall(t *testing.T) {
    tridiagSmall(t, false) 
}

func _TestTridiagUnitSmall(t *testing.T) {
    tridiagSmall(t, true) 
}


func trmmTest(t *testing.T, A *matrix.FloatMatrix, flags Flags, nb int) bool {
    var B0 *matrix.FloatMatrix
    N := A.Cols()
    S := 0
    E := A.Cols()
    side := linalg.OptLeft
    if flags & RIGHT != 0 {
        B0 = matrix.FloatWithValue(2, A.Rows(), 2.0)
        side = linalg.OptRight
        E = B0.Rows()
    } else {
        B0 = matrix.FloatWithValue(A.Rows(), 2, 2.0)
        E = B0.Cols()
    }
    B1 := B0.Copy()

    trans := linalg.OptNoTrans
    if flags & TRANSA != 0 {
        trans = linalg.OptTransA
    }
    diag := linalg.OptNonUnit
    if flags & UNIT != 0 {
        diag = linalg.OptUnit
    }
    uplo := linalg.OptUpper
    if flags & LOWER != 0 {
        uplo = linalg.OptLower
    }

    blas.TrmmFloat(A, B0, 1.0, uplo, diag, trans, side)
    if A.Rows() < 8 {
        //t.Logf("..A\n%v\n", A)
        t.Logf("  BLAS B0:\n%v\n", B0)
    }

    Ar := A.FloatArray()
    Br := B1.FloatArray()
    if nb != 0 {
        DTrmmBlk(Br, Ar, 1.0, flags, B1.LeadingIndex(), A.LeadingIndex(),
            N, S, E, nb)
    } else {
        DTrmmUnblk(Br, Ar, 1.0, flags, B1.LeadingIndex(), A.LeadingIndex(),
            N, S, E, 0)
    }
    result := B0.AllClose(B1)
    t.Logf("   B0 == B1: %v\n", result)
    if A.Rows() < 8 {
        t.Logf("  DTrmm B1:\n%v\n", B1)
    }
    return result
}


func TestTrmmUnblkSmall(t *testing.T) {
    //bN := 7
    Udata3 := [][]float64{
        []float64{1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0}}

    Udata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0},
        []float64{0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0}}
    U := matrix.FloatMatrixFromTable(Udata, matrix.RowOrder)
    U3 := matrix.FloatMatrixFromTable(Udata3, matrix.RowOrder)
    _ = U
    _ = U3

    Ldata3 := [][]float64{
        []float64{1.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0},
        []float64{1.0, 2.0, 3.0}}

    Ldata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    L := matrix.FloatMatrixFromTable(Ldata, matrix.RowOrder)
    L3 := matrix.FloatMatrixFromTable(Ldata3, matrix.RowOrder)
    _ = L

    t.Logf("-- TRMM-UPPER, NON-UNIT ---")
    fail(t, trmmTest(t, U3, UPPER, 0))
    t.Logf("-- TRMM-UPPER, UNIT ---")
    fail(t, trmmTest(t, U3, UPPER|UNIT, 0))
    t.Logf("-- TRMM-UPPER, NON-UNIT, TRANSA ---")
    fail(t, trmmTest(t, U3, UPPER|TRANSA, 0))
    t.Logf("-- TRMM-UPPER, UNIT, TRANSA ---")
    fail(t, trmmTest(t, U3, UPPER|TRANSA|UNIT, 0))
    t.Logf("-- TRMM-LOWER, NON-UNIT ---")
    fail(t, trmmTest(t, L3, LOWER, 0))
    t.Logf("-- TRMM-LOWER, UNIT ---")
    fail(t, trmmTest(t, L3, LOWER|UNIT, 0))
    t.Logf("-- TRMM-LOWER, NON-UNIT, TRANSA ---")
    fail(t, trmmTest(t, L3, LOWER|TRANSA, 0))
    t.Logf("-- TRMM-LOWER, UNIT, TRANSA ---")
    fail(t, trmmTest(t, L3, LOWER|TRANSA|UNIT, 0))

    t.Logf("-- TRMM-UPPER, NON-UNIT, RIGHT ---")
    fail(t, trmmTest(t, U3, UPPER|RIGHT, 0))
    t.Logf("-- TRMM-UPPER, UNIT, RIGHT ---")
    fail(t, trmmTest(t, U3, UPPER|UNIT|RIGHT, 0))

    t.Logf("-- TRMM-LOWER, NON-UNIT, RIGHT ---")
    fail(t, trmmTest(t, L3, LOWER|RIGHT, 0))
    t.Logf("-- TRMM-LOWER, UNIT, RIGHT ---")
    fail(t, trmmTest(t, L3, LOWER|UNIT|RIGHT, 0))

    t.Logf("-- TRMM-UPPER, NON-UNIT, RIGHT, TRANSA ---")
    fail(t, trmmTest(t, U3, UPPER|RIGHT|TRANSA, 0))
    t.Logf("-- TRMM-UPPER, UNIT, RIGHT, TRANSA ---")
    fail(t, trmmTest(t, U3, UPPER|UNIT|RIGHT|TRANSA, 0))

    t.Logf("-- TRMM-LOWER, NON-UNIT, RIGHT, TRANSA ---")
    fail(t, trmmTest(t, L3, LOWER|RIGHT|TRANSA, 0))
    t.Logf("-- TRMM-LOWER, UNIT, RIGHT, TRANSA ---")
    fail(t, trmmTest(t, L3, LOWER|UNIT|RIGHT|TRANSA, 0))

}

func _TestTrmmBlkSmall(t *testing.T) {
    //bN := 7
    Udata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0},
        []float64{0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0}}
    U := matrix.FloatMatrixFromTable(Udata, matrix.RowOrder)
    _ = U

    Ldata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    L := matrix.FloatMatrixFromTable(Ldata, matrix.RowOrder)
    _ = L

    t.Logf("-- TRMM-UPPER, NON-UNIT ---")
    fail(t, trmmTest(t, U, UPPER, 2))
    t.Logf("-- TRMM-UPPER, NON-UNIT, TRANSA ---")
    fail(t, trmmTest(t, U, UPPER|TRANSA, 2))
    t.Logf("-- TRMM-LOWER, NON-UNIT ---")
    fail(t, trmmTest(t, L, LOWER, 2))
    t.Logf("-- TRMM-LOWER, NON-UNIT, TRANSA ---")
    fail(t, trmmTest(t, L, LOWER|TRANSA, 2))
    t.Logf("-- TRMM-UPPER, RIGHT, NON-UNIT ---")
    fail(t, trmmTest(t, U, UPPER|RIGHT, 2))
    t.Logf("-- TRMM-UPPER, RIGHT, NON-UNIT, TRANSA ---")
    fail(t, trmmTest(t, U, UPPER|RIGHT|TRANSA, 2))
    t.Logf("-- TRMM-LOWER, RIGHT, NON-UNIT ---")
    fail(t, trmmTest(t, U, LOWER|RIGHT, 2))
    t.Logf("-- TRMM-LOWER, RIGHT, NON-UNIT, TRANSA ---")
    fail(t, trmmTest(t, U, LOWER|RIGHT|TRANSA, 2))
}


func trsmSolve(t *testing.T, A *matrix.FloatMatrix, flags Flags, rand bool, nb int) bool {
    var B0 *matrix.FloatMatrix
    side := linalg.OptLeft
    trans := linalg.OptNoTrans
    N := A.Cols()
    S := 0
    E := A.Rows()
    _ = S; _ = E
    if flags & RIGHT != 0 {
        if rand {
            B0 = matrix.FloatNormal(2, A.Rows())
        } else {
            B0 = matrix.FloatWithValue(2, A.Rows(), 2.0)
        }
        side = linalg.OptRight
        E = B0.Rows()
    } else {
        if rand {
            B0 = matrix.FloatNormal(A.Rows(), 2)
        } else {
            B0 = matrix.FloatWithValue(A.Rows(), 2, 2.0)
        }
        E = B0.Cols()
    }
    B1 := B0.Copy()
    diag := linalg.OptNonUnit
    if flags & UNIT != 0 {
        diag = linalg.OptUnit
    }
    uplo := linalg.OptUpper
    if flags & LOWER != 0 {
        uplo = linalg.OptLower
    }
    if flags & TRANSA != 0 {
        trans = linalg.OptTransA
    }
    blas.TrsmFloat(A, B0, 1.0, uplo, diag, side, trans)
    if N < 8 {
        //t.Logf("A=\n%v\n", A)
        t.Logf("blas: B0\n%v\n", B0)
    }

    Ar := A.FloatArray()
    Br := B1.FloatArray()
    if nb == 0 || nb == N {
        DSolveUnblk(Br, Ar, 1.0, flags, B1.LeadingIndex(), A.LeadingIndex(), N, S, E)
    } else {
        DSolveBlk(Br, Ar, 1.0, flags, B1.LeadingIndex(), A.LeadingIndex(), N, S, E, nb)
    }
    result := B1.AllClose(B0)
    t.Logf("B1 == B0: %v\n", result)
    if N < 8 {
        t.Logf("B1:\n%v\n", B1)
    }
    return result
}


func _TestTrsmSmall(t *testing.T) {
    //bN := 7
    Udata3 := [][]float64{
        []float64{2.0, 2.0, 2.0},
        []float64{0.0, 3.0, 3.0},
        []float64{0.0, 0.0, 4.0}}

    Udata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0},
        []float64{0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0}}
    U := matrix.FloatMatrixFromTable(Udata, matrix.RowOrder)
    U3 := matrix.FloatMatrixFromTable(Udata3, matrix.RowOrder)
    _ = U
    _ = U3

    Ldata3 := [][]float64{
        []float64{1.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0},
        []float64{1.0, 2.0, 3.0}}

    Ldata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    L := matrix.FloatMatrixFromTable(Ldata, matrix.RowOrder)
    L3 := matrix.FloatMatrixFromTable(Ldata3, matrix.RowOrder)
    _ = L
    _ = L3

    t.Logf("-- TRSM-LOWER, NON-UNIT --")
    trsmSolve(t, L3, LOWER, false, 0)
    //trsmSolve(t, L, LOWER, false, 0)
    //trsmSolve(t, L, LOWER, true, 0)

    t.Logf("- TRSM-LOWER, UNIT --")
    trsmSolve(t, L3, LOWER|UNIT, false, 0)

    t.Logf("-- TRSM-UPPER, NON-UNIT --")
    trsmSolve(t, U3, UPPER, false, 0)
    //trsmSolve(t, U, UPPER, false, 0)
    //trsmSolve(t, U, UPPER, true, 0)

    t.Logf("-- TRSM-UPPER, UNIT --")
    trsmSolve(t, U3, UPPER|UNIT, false, 0)

    t.Logf("-- TRSM-UPPER, TRANS, NON_UNIT --")
    trsmSolve(t, U3, UPPER|TRANSA, false, 0)

    t.Logf("-- TRSM-UPPER, TRANS, UNIT --")
    trsmSolve(t, U3, UPPER|TRANSA|UNIT, false, 0)

    t.Logf("-- TRSM-LOWER, TRANS, NON-UNIT --")
    trsmSolve(t, L3, LOWER|TRANSA, false, 0)

    t.Logf("-- TRSM-LOWER, TRANS, UNIT --")
    trsmSolve(t, L3, LOWER|TRANSA|UNIT, false, 0)

    t.Logf("-- TRSM-UPPER, NON-UNIT, RIGHT ---")
    trsmSolve(t, U3, UPPER|RIGHT, false, 0)

    t.Logf("-- TRSM-UPPER, UNIT, RIGHT ---")
    trsmSolve(t, U3, UPPER|UNIT|RIGHT, false, 0)

    t.Logf("-- TRSM-LOWER, NON-UNIT, RIGHT ---")
    trsmSolve(t, L3, LOWER|RIGHT, false, 0)

    t.Logf("-- TRSM-LOWER, UNIT, RIGHT ---")
    trsmSolve(t, L3, LOWER|UNIT|RIGHT, false, 0)

    t.Logf("-- TRSM-UPPER, NON-UNIT, RIGHT, TRANS ---")
    trsmSolve(t, U3, UPPER|RIGHT|TRANSA, false, 0)

    t.Logf("-- TRSM-UPPER, UNIT, RIGHT, TRANS ---")
    trsmSolve(t, U3, UPPER|UNIT|RIGHT|TRANSA, false, 0)

    t.Logf("-- TRSM-LOWER, NON-UNIT, RIGHT, TRANS ---")
    trsmSolve(t, L3, LOWER|RIGHT|TRANSA, false, 0)

    t.Logf("-- TRSM-LOWER, UNIT, RIGHT, TRANS ---")
    trsmSolve(t, L3, LOWER|UNIT|RIGHT|TRANSA, false, 0)

    t.Logf("-- BLK TRSM-UPPER, NON-UNIT ---")
    //trsmSolve(t, U, UPPER, false, 2)
    trsmSolve(t, U, UPPER, true, 2)

    t.Logf("-- BLK TRSM-UPPER, UNIT ---")
    //trsmSolve(t, U, UPPER, false, 2)
    trsmSolve(t, U, UPPER|UNIT, true, 2)

    t.Logf("-- BLK TRSM-LOWER, NON-UNIT ---")
    //trsmSolve(t, L, LOWER, false, 2)
    trsmSolve(t, L, LOWER, true, 2)

    t.Logf("-- BLK TRSM-LOWER, UNIT ---")
    //trsmSolve(t, L, LOWER, false, 2)
    trsmSolve(t, L, LOWER|UNIT, true, 2)

    t.Logf("-- BLK TRSM-UPPER, NON-UNIT, TRANS ---")
    //trsmSolve(t, U, UPPER|TRANSA, false, 2)
    trsmSolve(t, U, UPPER|TRANSA, true, 2)
    
    t.Logf("-- BLK TRSM-LOWER, NON-UNIT, TRANS ---")
    //trsmSolve(t, L, LOWER|TRANSA, false, 2)
    trsmSolve(t, L, LOWER|TRANSA, true, 2)

    t.Logf("-- BLK TRSM-UPPER, NON-UNIT, RIGHT ---")
    trsmSolve(t, U, UPPER|RIGHT, true, 2)

    t.Logf("-- BLK TRSM-UPPER, UNIT, RIGHT ---")
    trsmSolve(t, U, UPPER|UNIT|RIGHT, true, 2)

    t.Logf("-- BLK TRSM-UPPER, NON-UNIT, RIGHT, TRANSA ---")
    trsmSolve(t, U, UPPER|RIGHT|TRANSA, true, 2)

    t.Logf("-- BLK TRSM-LOWER, NON-UNIT, RIGHT ---")
    trsmSolve(t, L, LOWER|RIGHT, true, 2)

    t.Logf("-- BLK TRSM-LOWER, UNIT, RIGHT ---")
    trsmSolve(t, L, LOWER|UNIT|RIGHT, true, 2)

    t.Logf("-- BLK TRSM-LOWER, NON-UNIT, RIGHT, TRANSA ---")
    trsmSolve(t, L, LOWER|RIGHT|TRANSA, true, 2)
}

func _TestTrsmUnblk(t *testing.T) {
    //bN := 7
    Udata3 := [][]float64{
        []float64{2.0, 2.0, 2.0},
        []float64{0.0, 3.0, 3.0},
        []float64{0.0, 0.0, 4.0}}
    U3 := matrix.FloatMatrixFromTable(Udata3, matrix.RowOrder)
    _ = U3

    Ldata3 := [][]float64{
        []float64{1.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0},
        []float64{1.0, 2.0, 3.0}}
    L3 := matrix.FloatMatrixFromTable(Ldata3, matrix.RowOrder)
    _ = L3

    t.Logf("-- TRSM-UPPER, TRANS, RIGHT, NON_UNIT --")
    trsmSolve(t, U3, UPPER|TRANSA|RIGHT, false, 0)
    t.Logf("-- TRSM-UPPER, TRANS, RIGHT, UNIT --")
    trsmSolve(t, U3, UPPER|TRANSA|UNIT|RIGHT, false, 0)

    t.Logf("-- TRSM-LOWER, TRANS, RIGHT, NON-UNIT --")
    trsmSolve(t, L3, LOWER|TRANSA|RIGHT, false, 0)
    t.Logf("-- TRSM-LOWER, TRANS, RIGHT, UNIT --")
    trsmSolve(t, L3, LOWER|TRANSA|UNIT|RIGHT, false, 0)

}

func syrkTest(t *testing.T, C, A *matrix.FloatMatrix, flags Flags, vlen, nb int) bool {
    //var B0 *matrix.FloatMatrix
    P := A.Cols()
    S := 0
    E := C.Rows()
    C0 := C.Copy()

    trans := linalg.OptNoTrans
    if flags & TRANSA != 0 {
        trans = linalg.OptTrans
        P = A.Rows()
    }
    uplo := linalg.OptUpper
    if flags & LOWER != 0 {
        uplo = linalg.OptLower
    }

    blas.SyrkFloat(A, C0, 1.0, 1.0, uplo, trans)
    if A.Rows() < 8 {
        //t.Logf("..A\n%v\n", A)
        t.Logf("  BLAS C0:\n%v\n", C0)
    }

    Ar := A.FloatArray()
    Cr := C.FloatArray()
    DSymmRankBlk(Cr, Ar, 1.0, 1.0, flags, C.LeadingIndex(), A.LeadingIndex(),
        P, S, E, vlen, nb)
    result := C0.AllClose(C)
    t.Logf("   C0 == C: %v\n", result)
    if A.Rows() < 8 {
        t.Logf("  DMRank C:\n%v\n", C)
    }
    return result
}

func TestSyrkSmall(t *testing.T) {
    //bN := 7
    Udata3 := [][]float64{
        []float64{2.0, 2.0, 2.0},
        []float64{0.0, 3.0, 3.0},
        []float64{0.0, 0.0, 4.0}}

    Udata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0},
        []float64{0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0}}
    U := matrix.FloatMatrixFromTable(Udata, matrix.RowOrder)
    U3 := matrix.FloatMatrixFromTable(Udata3, matrix.RowOrder)
    _ = U
    _ = U3

    Ldata3 := [][]float64{
        []float64{1.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0},
        []float64{1.0, 2.0, 3.0}}

    Ldata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    L := matrix.FloatMatrixFromTable(Ldata, matrix.RowOrder)
    L3 := matrix.FloatMatrixFromTable(Ldata3, matrix.RowOrder)
    _ = L
    _ = L3

    Adata := [][]float64{
        []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    A := matrix.FloatMatrixFromTable(Adata);
    t.Logf("-- SYRK UPPER --")
    syrkTest(t, U.Copy(), A, UPPER, 4, 2)
    t.Logf("-- SYRK LOWER --")
    syrkTest(t, L.Copy(), A, LOWER, 4, 2)
    t.Logf("-- SYRK UPPER, TRANSA --")
    t.Logf("A: \n%v\n", A.Transpose())
    syrkTest(t, U.Copy(), A.Transpose(), UPPER|TRANSA, 4, 2)
    t.Logf("-- SYRK LOWER --")
    syrkTest(t, L.Copy(), A.Transpose(), LOWER|TRANSA, 4, 2)
}

func syrk2Test(t *testing.T, C, A, B *matrix.FloatMatrix, flags Flags, vlen, nb int) bool {
    //var B0 *matrix.FloatMatrix
    P := A.Cols()
    S := 0
    E := C.Rows()
    C0 := C.Copy()

    trans := linalg.OptNoTrans
    if flags & TRANSA != 0 {
        trans = linalg.OptTrans
        P = A.Rows()
    }
    uplo := linalg.OptUpper
    if flags & LOWER != 0 {
        uplo = linalg.OptLower
    }

    blas.Syr2kFloat(A, B, C0, 1.0, 1.0, uplo, trans)
    if A.Rows() < 8 {
        //t.Logf("..A\n%v\n", A)
        t.Logf("  BLAS C0:\n%v\n", C0)
    }

    Ar := A.FloatArray()
    Br := B.FloatArray()
    Cr := C.FloatArray()
    DSymmRank2Blk(Cr, Ar, Br, 1.0, 1.0, flags, C.LeadingIndex(), A.LeadingIndex(),
        B.LeadingIndex(), P, S, E, vlen, nb)
    result := C0.AllClose(C)
    t.Logf("   C0 == C: %v\n", result)
    if A.Rows() < 8 {
        t.Logf("  DMRank2 C:\n%v\n", C)
    }
    return result
}

func TestSyrk2Small(t *testing.T) {
    //bN := 7
    Udata3 := [][]float64{
        []float64{2.0, 2.0, 2.0},
        []float64{0.0, 3.0, 3.0},
        []float64{0.0, 0.0, 4.0}}

    Udata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0},
        []float64{0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0},
        []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0}}
    U := matrix.FloatMatrixFromTable(Udata, matrix.RowOrder)
    U3 := matrix.FloatMatrixFromTable(Udata3, matrix.RowOrder)
    _ = U
    _ = U3

    Ldata3 := [][]float64{
        []float64{1.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0},
        []float64{1.0, 2.0, 3.0}}

    Ldata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    L := matrix.FloatMatrixFromTable(Ldata, matrix.RowOrder)
    L3 := matrix.FloatMatrixFromTable(Ldata3, matrix.RowOrder)
    _ = L
    _ = L3

    Adata := [][]float64{
        []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}
    Bdata := [][]float64{
        []float64{7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
        []float64{7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}}
    _ = Bdata
    A := matrix.FloatMatrixFromTable(Adata);
    //B := matrix.FloatMatrixFromTable(Bdata);
    B := matrix.FloatNormal(7, 2);
    t.Logf("-- SYR2K UPPER --")
    syrk2Test(t, U.Copy(), A, B, UPPER, 4, 2)
    t.Logf("-- SYR2K LOWER --")
    syrk2Test(t, L.Copy(), A, B, LOWER, 4, 2)
    t.Logf("-- SYR2K UPPER, TRANSA --")
    //t.Logf("A: \n%v\n", A.Transpose())
    syrk2Test(t, U.Copy(), A.Transpose(), B.Transpose(), UPPER|TRANSA, 4, 2)
    t.Logf("-- SYR2K LOWER, TRANS --")
    syrk2Test(t, L.Copy(), A.Transpose(), B.Transpose(), LOWER|TRANSA, 4, 2)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
