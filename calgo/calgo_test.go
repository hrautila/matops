
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


func TestMakeData(t *testing.T) {
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
    E := matrix.FloatMatrixFromTable(Ddata, matrix.RowOrder)
     */
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

    DMult(C1r, Dr, Er, 1.0, 1.0, NOTRANS, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}


func TestMultTransASmall(t *testing.T) {
    bM := 5
    bN := 5
    bP := 5
    /*
    Ddata := [][]float64{
        []float64{1.0, 1.0, 1.0, 1.0, 1.0},
        []float64{2.0, 2.0, 2.0, 2.0, 2.0},
        []float64{3.0, 3.0, 3.0, 3.0, 3.0},
        []float64{4.0, 4.0, 4.0, 4.0, 4.0},
        []float64{5.0, 5.0, 5.0, 5.0, 5.0}}
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

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSA, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func TestMultTransBSmall(t *testing.T) {
    bM := 5
    bN := 5
    bP := 5
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

    DMult(C1r, Dr, Er, 1.0, 1.0, TRANSB, bM, bM, bP, bP, 0,  bN, 0,  bM, 32, 32, 32)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
}

func TestMultTransABSmall(t *testing.T) {
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

func TestMultMVSmall(t *testing.T) {
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

func TestMultMV(t *testing.T) {
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

func TestMultMVTransASmall(t *testing.T) {
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
    bN := 5
    bP := 5
    Adata := [][]float64{
     []float64{1.0, 1.0, 1.0, 1.0, 1.0},
     []float64{0.0, 2.0, 2.0, 2.0, 2.0},
     []float64{0.0, 0.0, 3.0, 3.0, 3.0},
     []float64{0.0, 0.0, 0.0, 4.0, 4.0},
     []float64{0.0, 0.0, 0.0, 0.0, 5.0}}

    //A := matrix.FloatNormal(bN, bN)
    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    //B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    B := matrix.FloatWithValue(bN, bP, 1.0)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    t.Logf("A=\n%v\n", A)
    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptUpper)
    t.Logf("blas: C=A*B\n%v\n", C0)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, UPPER|LEFT, bN, A.LeadingIndex(), bN, bN, 0,  bP, 0,  bN, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)
}

func _TestMultSymmLowerSmall(t *testing.T) {
    //bM := 5
    bN := 5
    bP := 5
    Adata := [][]float64{
     []float64{1.0, 0.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 0.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 0.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 0.0},
     []float64{1.0, 2.0, 3.0, 4.0, 5.0}}

    //A := matrix.FloatNormal(bN, bN)
    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    //B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    B := matrix.FloatWithValue(bN, bP, 1.0)
    C0 := matrix.FloatZeros(bN, bP)
    C1 := matrix.FloatZeros(bN, bP)

    Ar := A.FloatArray()
    Br := B.FloatArray()
    C1r := C1.FloatArray()

    t.Logf("A=\n%v\n", A)
    blas.SymmFloat(A, B, C0, 1.0, 1.0, linalg.OptLower)
    t.Logf("blas: C=A*B\n%v\n", C0)

    DMultSymm(C1r, Ar, Br, 1.0, 1.0, LOWER|LEFT, bN, A.LeadingIndex(), bN,
        bN, 0,  bP, 0,  bN, 4, 4, 4)
    t.Logf("C0 == C1: %v\n", C0.AllClose(C1))
    t.Logf("C1: C1 = A*X\n%v\n", C1)
}

func _TestMultSymmUpper(t *testing.T) {
    //bM := 5
    bN := 100*N
    bP := 100*P
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

func TestMultSyrSmall(t *testing.T) {
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

func TestMultSyr2Small(t *testing.T) {
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

func TestSolveForwardSmall(t *testing.T) {
    //bM := 5
    bN := 5
    Adata := [][]float64{
        []float64{1.0, 0.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0}}

    //A := matrix.FloatNormal(bN, bN)
    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    //B := matrix.FloatNormal(bN, bP)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    Z := matrix.FloatNormal(bN, 1);
    X0 := matrix.FloatWithValue(bN, 1, 0.0)
    X2 := matrix.FloatWithValue(bN, 1, 0.0)
    xsum := 0.0
    for i := 0; i < bN; i++ {
        xsum += float64(i) + 1.0
        X0.Add(xsum, i)
        X2.Add(xsum, -(i+1))
    }
    X0.Mul(Z)
    X1 := X0.Copy()
    X2.Mul(Z)
    X3 := X2.Copy()
    At := A.Transpose()
    Ar := A.FloatArray()
    Xr := X1.FloatArray()

    t.Logf("X0=\n%v\n", X0)
    t.Logf("Z=\n%v\n", Z)
    blas.TrsvFloat(A, X0, linalg.OptLower)
    t.Logf("blas: X0\n%v\n", X0)

    DSolveFwd(Xr, Ar, 1, A.LeadingIndex(), bN, bN)
    t.Logf("X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("X1:\n%v\n", X1)

    t.Logf("X2=\n%v\n", X2)
    //t.Logf("At=\n%v\n", At)
    blas.TrsvFloat(At, X2, linalg.OptUpper)
    t.Logf("blas: X2\n%v\n", X2)

    Xr = X3.FloatArray()
    Ar = At.FloatArray()
    DSolveBackwd(Xr, Ar, 1, At.LeadingIndex(), bN, bN)
    t.Logf("X2 == X1: %v\n", X2.AllClose(X3))
    t.Logf("X3:\n%v\n", X3)

}

func TestTridiagSmall(t *testing.T) {
    //bM := 5
    bN := 5
    Adata := [][]float64{
        []float64{1.0, 0.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 0.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 0.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 0.0},
        []float64{1.0, 2.0, 3.0, 4.0, 5.0}}

    //A := matrix.FloatNormal(bN, bN)
    A := matrix.FloatMatrixFromTable(Adata, matrix.RowOrder)
    //A := matrix.FloatWithValue(bM, bP, 2.0)
    //Z := matrix.FloatNormal(bN, 1);
    X0 := matrix.FloatWithValue(bN, 1, 1.0)
    X2 := matrix.FloatWithValue(bN, 1, 1.0)
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
    At := A.Transpose()

    t.Logf("X0=\n%v\n", X0)
    t.Logf("A.t=\n%v\n", At)
    t.Logf("A=\n%v\n", A)
    //t.Logf("Z=\n%v\n", Z)
    blas.TrmvFloat(A, X0, linalg.OptUpper)
    t.Logf("blas(upper): X0 = A*X0\n%v\n", X0)

    Ar := A.FloatArray()
    Xr := X1.FloatArray()
    DTridiagFwd(Xr, Ar, 1, At.LeadingIndex(), bN, bN)
    t.Logf("X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("X1(fwd) = A*X1:\n%v\n", X1)
    
    X0.SetIndexes(1.0)
    X1.SetIndexes(1.0)

    blas.TrmvFloat(At, X0, linalg.OptUpper)
    t.Logf("blas(upper): X0 = A.t*X0\n%v\n", X0)

    Ar = At.FloatArray()
    Xr = X1.FloatArray()
    DTridiagFwd(Xr, Ar, 1, At.LeadingIndex(), bN, bN)
    t.Logf("X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("X1(fwd) = A.t*X1:\n%v\n", X1)

    X0.SetIndexes(1.0)
    X1.SetIndexes(1.0)
    blas.TrmvFloat(A, X0, linalg.OptLower)
    t.Logf("blas(lower): X0 = A*X0\n%v\n", X0)

    Ar = A.FloatArray()
    Xr = X1.FloatArray()
    DTridiagBackwd(Xr, Ar, 1, At.LeadingIndex(), bN, bN)
    t.Logf("X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("X1(backwd) = A*X1:\n%v\n", X1)

    X0.SetIndexes(1.0)
    X1.SetIndexes(1.0)
    blas.TrmvFloat(At, X0, linalg.OptLower)
    t.Logf("blas(lower): X0 = A.t*X0\n%v\n", X0)

    Ar = At.FloatArray()
    Xr = X1.FloatArray()
    DTridiagBackwd(Xr, Ar, 1, At.LeadingIndex(), bN, bN)
    t.Logf("X0 == X1: %v\n", X0.AllClose(X1))
    t.Logf("X1(backwd) = A.t*X1:\n%v\n", X1)

/*
    t.Logf("X2=\n%v\n", X2)
    //t.Logf("At=\n%v\n", At)
    blas.TrsvFloat(At, X2, linalg.OptUpper)
    t.Logf("blas: X2\n%v\n", X2)

    Xr = X3.FloatArray()
    Ar = At.FloatArray()
    DSolveBackwd(Xr, Ar, 1, At.LeadingIndex(), bN, bN)
    t.Logf("X2 == X1: %v\n", X2.AllClose(X3))
    t.Logf("X3:\n%v\n", X3)
*/
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
