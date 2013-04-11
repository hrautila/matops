
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    //"fmt"
)

var decompNB int = 0

func DecomposeBlockSize(nb int) {
    decompNB = nb
}

func unblockedLUnoPiv(A *matrix.FloatMatrix) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 matrix.FloatMatrix

    err = nil
    partition2x2(&ATL, &ATR, &ABL, &ABR, A, 0)
    for ATL.Rows() < A.Rows() {
        repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, A, 1)

        // a21 = a21/a11
        a21.Scale(1.0/a11.Float())
        // A22 = A22 - a21*a12
        err = MVRankUpdate(&A22, &a21, &a12, -1.0)

        continue3x3to2x2(&ATL, &ATR, &ABL, &ABR, &A00, &a11, &A22, A)
    }
    return
}

func m(A *matrix.FloatMatrix) int {
    return A.Rows()
}

func blockedLUnoPiv(A *matrix.FloatMatrix, nb int) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 matrix.FloatMatrix

    err = nil
    partition2x2(&ATL, &ATR, &ABL, &ABR, A, 0)
    for ATL.Rows() < A.Rows() {
        repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22, A, nb)

        // A00 = LU(A00)
        unblockedLUnoPiv(&A11)
        // A12 = trilu(A00)*A12.-1  (TRSM)
        Solve(&A12, &A11, 1.0, LEFT|LOWER|UNIT)
        // A21 = A21.-1*triu(A00) (TRSM)
        Solve(&A21, &A11, 1.0, RIGHT|UPPER)
        // A22 = A22 - A21*A12
        Mult(&A22, &A21, &A12, -1.0, 1.0, NOTRANS)

        continue3x3to2x2(&ATL, &ATR, &ABL, &ABR, &A00, &A11, &A22, A)
    }
    return
}

func DecomposeLUnoPiv(A *matrix.FloatMatrix, nb int) (*matrix.FloatMatrix, error) {
    var err error
    if nb == 0 {
        err = unblockedLUnoPiv(A)
    } else {
        err = blockedLUnoPiv(A, nb)
    }
    return A, err
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
