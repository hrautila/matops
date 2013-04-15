
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "errors"
    //"fmt"
)

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

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
        //a21.Scale(1.0/a11.Float())
        InvScale(&a21, a11.Float())
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

func DecomposeLUnoPiv(A *matrix.FloatMatrix) (*matrix.FloatMatrix, error) {
    var err error
    mlen := min(A.Rows(), A.Cols())
    if mlen <= decompNB || decompNB == 0 {
        err = unblockedLUnoPiv(A)
    } else {
        err = blockedLUnoPiv(A, decompNB)
    }
    return A, err
}

func swapRows(A *matrix.FloatMatrix, src, dst int) {
    var r0, r1 matrix.FloatMatrix
    if src == dst || A.Rows() == 0 {
        return
    }
    r0.SubMatrixOf(A, src, 0, 1, A.Cols())
    r1.SubMatrixOf(A, dst, 0, 1, A.Cols())
    Swap(&r0, &r1)
    /*
    for k := 0; k < r0.Cols(); k++ {
        tmp := r0.GetAt(0, k)
        r0.SetAt(0, k, r1.GetAt(0, k))
        r1.SetAt(0, k, tmp)
    }
     */
}

func applyPivots(A *matrix.FloatMatrix, p *pPivots) {
    for k, n := range p.pivots {
        if n > 0 {
            swapRows(A, n, k)
        }
    }
}

func pivotIndex(A *matrix.FloatMatrix, p *pPivots) {
    max := A.GetAt(0, 0)
    for k := 1; k < A.Rows(); k++ {
        v := A.GetAt(k, 0)
        if v != 0 && (v > max || max == 0.0) {
            p.pivots[0] = k
            max = v
        }
    }
}

func VDot(X, Y *matrix.FloatMatrix) float64 {
    var rval float64 = 0.0
    if ! (isVector(X) || isVector(Y)) {
        return rval
    }
    for i := 0; i < X.NumElements(); i++ {
        rval += X.GetAt(0, i)*Y.GetAt(i, 0)
    }
    return rval
}

func unblockedLUpiv(A *matrix.FloatMatrix, p *pPivots) error {
    var err error
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 matrix.FloatMatrix
    var AL, AR, A0, a1, A2, aB1, AB0 matrix.FloatMatrix
    var pT, pB, p0, p1, p2 pPivots

    err = nil
    partition2x2(&ATL, &ATR, &ABL, &ABR, A, 0)
    partition1x2(&AL, &AR, A, 0, pRIGHT)
    partitionPivot2x1(&pT, &pB, p, 0, pBOTTOM)

    for ATL.Rows() < A.Rows() && ATL.Cols() < A.Cols() {
        repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, A, 1)
        repartition1x2to1x3(&AL, &A0, &a1, &A2, A, 1, pRIGHT)
        repartPivot2x1to3x1(&pT, &p0, &p1, &p2, p, 1, pBOTTOM)

        // apply previously computed pivots
        applyPivots(&a1, &p0)

        // a01 = trilu(A00) \ a01 (TRSV)
        MVSolve(&a01, &A00, 1.0, LOWER|UNIT)
        a11.Add(-VDot(&a10, &a01))
        MVMult(&a21, &A20, &a01, -1.0, 1.0)

        // pivot index on current column [a11, a21].T
        aB1.SubMatrixOf(&ABR, 0, 0, ABR.Rows(), 1)
        pivotIndex(&aB1, &p1)

        // pivots to current column
        applyPivots(&aB1, &p1)
        
        // a21 = a21 / a11
        a21.Scale(1.0/a11.Float())

        // apply pivots to previous columns
        AB0.SubMatrixOf(&ABL, 0, 0)
        applyPivots(&AB0, &p1)
        // scale last pivots to origin matrix row numbers
        p1.pivots[0] += ATL.Rows()

        continue3x3to2x2(&ATL, &ATR, &ABL, &ABR, &A00, &a11, &A22, A)
        continue1x3to1x2(&AL, &AR, &A0, &a1, A, pRIGHT)
        contPivot3x1to2x1(&pT, &pB, &p0, &p1, p, pBOTTOM)
    }
    if ATL.Cols() < A.Cols() {
        //AB0.SubMatrixOf(A, 0, ATL.Cols())
        applyPivots(&ATR, p)
        Solve(&ATR, &ATL, 1.0, LEFT|UNIT|LOWER)
    }
    return err
}

func blockedLUpiv(A *matrix.FloatMatrix, p *pPivots, nb int) error {
    var err error
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 matrix.FloatMatrix
    var AL, AR, A0, A1, A2, AB1, AB0 matrix.FloatMatrix
    var pT, pB, p0, p1, p2 pPivots

    err = nil
    partition2x2(&ATL, &ATR, &ABL, &ABR, A, 0)
    partition1x2(&AL, &AR, A, 0, pRIGHT)
    partitionPivot2x1(&pT, &pB, p, 0, pBOTTOM)

    for ATL.Rows() < A.Rows() && ATL.Cols() < A.Cols() {
        repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22, A, nb)
        repartition1x2to1x3(&AL, &A0, &A1, &A2, A, nb, pRIGHT)
        repartPivot2x1to3x1(&pT, &p0, &p1, &p2, p, nb, pBOTTOM)
        // apply previously computed pivots
        applyPivots(&A1, &p0)

        // a01 = trilu(A00) \ a01 (TRSV)
        Solve(&A01, &A00, 1.0, LOWER|UNIT)
        // A11 = A11 - A10*A01
        Mult(&A11, &A10, &A01, -1.0, 1.0, NOTRANS)
        // A21 = A21 - A20*A01
        Mult(&A21, &A20, &A01, -1.0, 1.0, NOTRANS)

        // LU_piv(AB1, p1)
        AB1.SubMatrixOf(&ABR, 0, 0, ABR.Rows(), A11.Cols())
        unblockedLUpiv(&AB1, &p1)

        // apply pivots to previous columns
        AB0.SubMatrixOf(&ABL, 0, 0)
        applyPivots(&AB0, &p1)
        // scale last pivots to origin matrix row numbers
        for k, _ := range p1.pivots {
            p1.pivots[k] += ATL.Rows()
        }

        continue3x3to2x2(&ATL, &ATR, &ABL, &ABR, &A00, &A11, &A22, A)
        continue1x3to1x2(&AL, &AR, &A0, &A1, A, pRIGHT)
        contPivot3x1to2x1(&pT, &pB, &p0, &p1, p, pBOTTOM)
    }
    if ATL.Cols() < A.Cols() {
        applyPivots(&ATR, p)
        Solve(&ATR, &ATL, 1.0, LEFT|UNIT|LOWER)
    }
    return err
}

func DecomposeLU(A *matrix.FloatMatrix, pivots []int) (*matrix.FloatMatrix, error) {
    var err error
    mlen := min(A.Rows(), A.Cols())
    if len(pivots) < mlen {
        return A, errors.New("pivot array < min(A.Rows(),A.Cols())")
    }
    // clear pivot array
    for k, _ := range pivots {
        pivots[k] = 0
    }
    if mlen <= decompNB || decompNB == 0 {
        err = unblockedLUpiv(A, &pPivots{pivots})
    } else {
        err = blockedLUpiv(A, &pPivots{pivots}, decompNB)
    }
    return A, err
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
