
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    //"math"
    //"fmt"
)

// unblocked QR; compatible to lapack.DGEQRF
func unblockedQR(A, T *matrix.FloatMatrix, mlen int) {
    var As, ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 matrix.FloatMatrix
    var TT, TB matrix.FloatMatrix
    var t0, tau1, t2  matrix.FloatMatrix

    As.SubMatrixOf(A, 0, 0, mlen, mlen)
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, &As, 0, pTOPLEFT)
    partition2x1(
        &TT,
        &TB,  T, 0, pTOP)

    for ABR.Rows() > 0 && ABR.Cols() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22,   &As, 1, pBOTTOMRIGHT)
        repartition2x1to3x1(&TT,
            &t0,
            &tau1,
            &t2,     T, 1, pBOTTOM)

        computeHouseholder(&a11, &a21, &tau1, LEFT)
        applyHouseholder(&tau1, &a21, &a12, &A22, LEFT)
        
        //householderUT(&a11, &a21, &tau1, LEFT)
        //householderApplyUT(&tau1, &a21, &a12, &A22, LEFT)

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   &As, pBOTTOMRIGHT)

        continue3x1to2x1(
            &TT,
            &TB,   &t0, &tau1,   T, pBOTTOM)
    }
}

// Compute QR decompostion of matrix A. 
func DecomposeQR(A, tau *matrix.FloatMatrix) (*matrix.FloatMatrix, error) {
    var err error = nil
    mlen := A.Rows()
    if mlen > A.Cols() {
        mlen = A.Cols()
    }
    unblockedQR(A, tau, mlen)
    return A, err
}


// not working correctly, yet! 
func unblockedQRUT(A, T *matrix.FloatMatrix) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 matrix.FloatMatrix
    var TTL, TTR, TBL, TBR matrix.FloatMatrix
    var T00, t01, T02, t10, tau11, t12, T20, t21, T22 matrix.FloatMatrix

    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, pTOPLEFT)
    partition2x2(
        &TTL, &TTR,
        &TBL, &TBR, T, 0, pTOPLEFT)

    for ABR.Rows() > 0 && ABR.Cols() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22,   A, 1, pBOTTOMRIGHT)
        repartition2x2to3x3(&TTL,
            &T00, &t01,   &T02,
            &t10, &tau11, &t12,
            &T20, &t21,   &T22,   T, 1, pBOTTOMRIGHT)

        // 
        computeHouseholder(&a11, &a21, &tau11, LEFT)
        applyHouseholder(&tau11, &a21, &a12, &A22, LEFT)

        // t01 = a10 + A20 * a21
        a10.CopyTo(&t01)
        MVMult(&t01, &A20, &a21, 1.0, 1.0, NOTRANS)

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, pBOTTOMRIGHT)

        continue3x3to2x2(
            &TTL, &TTR,
            &TBL, &TBR,   &T00, &tau11, &T22,   T, pBOTTOMRIGHT)
    }
}


func DecomposeQRUT(A, T *matrix.FloatMatrix) (*matrix.FloatMatrix, error) {
    var err error = nil
    unblockedQRUT(A, T)
    return A, err
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
