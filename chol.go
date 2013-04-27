
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "errors"
    //"math"
    //"fmt"
)

func unblockedCHOL(A *matrix.FloatMatrix, flags Flags) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 matrix.FloatMatrix

    err = nil
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, pTOPLEFT)

    for ATL.Rows() < A.Rows() {
        repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22,   A, 1, pBOTTOMRIGHT)

        // a11 = sqrt(a11)
        a11.Sqrt()

        if flags & LOWER != 0 {
            // a21 = a21/a11
            InvScale(&a21, a11.Float())
            // A22 = A22 - a21*a21' (SYR)
            err = MVRankUpdateSym(&A22, &a21, -1.0, flags)
        } else {
            // a21 = a12/a11
            InvScale(&a12, a11.Float())
            // A22 = A22 - a12'*a12 (SYR)
            err = MVRankUpdateSym(&A22, &a12, -1.0, flags)
        }

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,  A, pBOTTOMRIGHT)
    }
    return
}

func blockedCHOL(A *matrix.FloatMatrix, flags Flags, nb int) error {
    var err error
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 matrix.FloatMatrix

    err = nil
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, pTOPLEFT)

    for ATL.Rows() < A.Rows() && ATL.Cols() < A.Cols() {
        repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22,   A, nb, pBOTTOMRIGHT)

        // A11 = chol(A11)
        err = unblockedCHOL(&A11, flags)

        if flags & LOWER != 0 {
            // A21 = A21 * tril(A11).-1
            Solve(&A21, &A11, 1.0, RIGHT|LOWER|TRANSA)
            // A22 = A22 - A21*A21.T
            RankUpdateSym(&A22, &A21, -1.0, 1.0, LOWER)
        } else {
            // A12 = triu(A11).-1 * A21
            Solve(&A12, &A11, 1.0, LEFT|UPPER|TRANSA)
            // A22 = A22 - A12*A12.T
            RankUpdateSym(&A22, &A12, -1.0, 1.0, UPPER|TRANSA)
        }

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pBOTTOMRIGHT)
    }
    return err
}

func DecomposeCHOL(A *matrix.FloatMatrix, flags Flags) (*matrix.FloatMatrix, error) {
    var err error
    if A.Cols() != A.Rows() {
        return A, errors.New("A not a square matrix")
    }
    if A.Cols() < decompNB || decompNB == 0 {
        err = unblockedCHOL(A, flags)
    } else {
        err = blockedCHOL(A, flags, decompNB)
    }
    return A, err
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
