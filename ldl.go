
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
	"github.com/hrautila/matrix"
    "errors"
    //"fmt"
)


/*
 *  ( a11  a12 )   ( 1   0   )( d1  0   )( l  l21.t )
 *  ( a21  A22 )   ( l21 L22 )(  0  A22 )( 0  L22.t )
 *
 *   a11  =   d1
 *   a21  =   l21*d1                       => l21 = a21/d1
 *   A22  =   l21*d1*l21.t + L22*D2*L22.t  => L22 = A22 - l21*d1*l21t
 */
func unblkLowerLDL(A *matrix.FloatMatrix, p *pPivots) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a10, a11,  A20, a21, A22, acol matrix.FloatMatrix
    var pT, pB, p0, p1, p2 pPivots

    err = nil
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, pTOPLEFT)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, pTOP)

    for ATL.Rows() < A.Rows() {
        repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, pBOTTOMRIGHT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,   /**/ p, 1, pBOTTOM)

        // --------------------------------------------------------

        merge2x1(&acol, &a11, &a21)
        imax := IAMax(&acol)
        if imax > 0 {
            // pivot diagonal in symmetric matrix; will swap [0,0] and [imax,imax]
            // We loose if these is zero on diagonal!! 
            applyPivotSym(&ABL, &ABR, imax, LOWER)
            p1.pivots[0] = imax + ATL.Rows() + 1
        } else {
            p1.pivots[0] = 0
        }
        // d11 = a11; no-op

        // A22 = A22 - l21*d11*l21.T = A22 - a21*a21.T/a11; triangular update
        err = MVUpdateTrm(&A22, &a21, &a21, -1.0/a11.Float(), LOWER)

        // l21 = a21/a11
        InvScale(&a21, a11.Float())
        // ---------------------------------------------------------

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,  A, pBOTTOMRIGHT)
        contPivot3x1to2x1(
            &pT,
            &pB,    &p0, &p1,    p, pBOTTOM)
    }
    return
}

/*
 *  ( A11  A12 )   ( L11   0  )( D1  0  )( L11.t  L21.t )
 *  ( A21  A22 )   ( L21  L22 )(  0  D2 )(   0    L22.t )
 *
 *   A11  =   L11*D1*L11.t                 -> L11\D1 = LDL(A11)
 *   A12  =   L11*D1*L21.t  
 *   A21  =   L21*D1*L11.t                 => L21 = A21*(D1*L11.t).-1 = A21*L11.-T*D1.-1
 *   A22  =   L21*D1*L21.t + L22*D2*L22.t  => L22 = A22 - L21*D1*L21.t
 */
func blkLowerLDL(A, W *matrix.FloatMatrix, p *pPivots, nb int) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, A10, A11,  A20, A21, A22 matrix.FloatMatrix
    var D1, wrk matrix.FloatMatrix
    var pT, pB, p0, p1, p2 pPivots

    err = nil
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, pTOPLEFT)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, pTOP)

    for ATL.Rows() < A.Rows() {
        repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pBOTTOMRIGHT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,   /**/ p, nb, pBOTTOM)

        // --------------------------------------------------------

        // A11 = LDL(A11)
        unblkLowerLDL(&A11, &p1)
        applyRowPivots(&A10, &p1, 0, FORWARD)
        applyColPivots(&A21, &p1, 0, FORWARD)
        scalePivots(&p1, ATL.Rows())
        
        A11.Diag(&D1)
        // A21 = A21*A11.-T
        SolveTrm(&A21, &A11, 1.0, LOWER|UNIT|RIGHT|TRANSA)
        // A21 = A21*D1.-1
        SolveDiag(&A21, &D1, RIGHT)

        // W = D1*L21.T = L21*D1
        W.SubMatrix(&wrk, 0, 0, A21.Rows(), nb)
        A21.CopyTo(&wrk)
        MultDiag(&wrk, &D1, RIGHT)

        // A22 = A22 - L21*D1*L21.T = A22 - L21*W.T
        UpdateTrm(&A22, &A21, &wrk, -1.0, 1.0, LOWER|TRANSB)

        // ---------------------------------------------------------

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,  A, pBOTTOMRIGHT)
        contPivot3x1to2x1(
            &pT,
             &pB,    &p0, &p1,    p, pBOTTOM)
    }
    return
}

/*
 *  ( A11  a12 )   ( U11 u12 )( D1  0  )( U11.t 0 )
 *  ( a21  a22 )   (  0   1  )(  0  d2 )( u12.t 1 )
 *
 *   a22  =   d2
 *   a01  =   u12*d2                       => u12 = a12/d2
 *   A11  =   u12*d2*u12.t + U11*D1*U11.t  => U11 = A11 - u12*d2*u12.t
 */
func unblkUpperLDL(A *matrix.FloatMatrix, p *pPivots) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a01, A02, a11, a12, A22 matrix.FloatMatrix
    var AL, AR, acol matrix.FloatMatrix
    var pT, pB, p0, p1, p2 pPivots

    err = nil
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, pBOTTOMRIGHT)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, pBOTTOM)

    for ATL.Rows() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            nil,  &a11, &a12,
            nil,  nil,  &A22,   A, 1, pTOPLEFT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,   /**/ p, 1, pTOP)

        // --------------------------------------------------------
        merge2x1(&acol, &a01, &a11)
        imax := IAMax(&acol)
        if imax < acol.NumElements()-1 {
            merge1x2(&AL, &ATL, &ATR)
            merge1x2(&AR, &a11, &a12)
            // pivot diagonal in symmetric matrix; will swap [0,0] and [imax,imax]
            // We loose if there is zero on diagonal!! 
            applyPivotSym(&AL, &AR, imax, UPPER)
            p1.pivots[0] = imax + 1
        } else {
            p1.pivots[0] = 0
        }

        // A00 = A00 - u01*d11*u01.T = A00 - a01*a01.T/a11; triangular update
        err = MVUpdateTrm(&A00, &a01, &a01, -1.0/a11.Float(), UPPER)

        // u01 = a01/a11
        InvScale(&a01, a11.Float())
        // ---------------------------------------------------------

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,  A, pTOPLEFT)
        contPivot3x1to2x1(
            &pT,
            &pB,    &p0, &p1,    p, pTOP)
    }
    return
}


func blkUpperLDL(A, W *matrix.FloatMatrix, p *pPivots, nb int) (err error) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, A01, A02, A11, A12, A22 matrix.FloatMatrix
    var D1, wrk matrix.FloatMatrix
    var pT, pB, p0, p1, p2 pPivots

    err = nil
    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, pBOTTOMRIGHT)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, pBOTTOM)

    for ATL.Rows() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            nil,  &A11, &A12,
            nil,  nil,  &A22,   A, nb, pTOPLEFT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,   /**/ p, nb, pTOP)

        // --------------------------------------------------------

        // A11 = LDL(A11)
        unblkUpperLDL(&A11, &p1)
        applyColPivots(&A01, &p1, 0, BACKWARD)
        applyRowPivots(&A12, &p1, 0, BACKWARD)
        scalePivots(&p1, ATL.Rows()-A11.Rows())

        A11.Diag(&D1)

        // A01 = A01*A11.-T
        SolveTrm(&A01, &A11, 1.0, UPPER|UNIT|RIGHT|TRANSA)
        // A01 = A01*D1.-1
        SolveDiag(&A01, &D1, RIGHT)

        // W = D1*U01.T = U01*D1
        W.SubMatrix(&wrk, 0, 0, A01.Rows(), nb)
        A01.CopyTo(&wrk)
        MultDiag(&wrk, &D1, RIGHT)

        // A00 = A00 - U01*D1*U01.T = A22 - U01*W.T
        UpdateTrm(&A00, &A01, &wrk, -1.0, 1.0, UPPER|TRANSB)

        // ---------------------------------------------------------

        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,  A, pTOPLEFT)
        contPivot3x1to2x1(
            &pT,
            &pB,    &p0, &p1,    p, pTOP)
    }
    return
}

/*
 * Compute an LDLT factorization of a symmetric N-by-N matrix with partial pivoting.
 *
 * Arguments:
 *   A      On entry, the N-by-N matrix to be factored. On exit the factor
 *          L and 1-by-1 diagonal D from factorization A = L*D*L.T, the unit diagonal 
 *          of L are not stored. Or the factor U and diagonal D from factorization
 *          A = U*D*U.T if flag bit UPPER is set.
 *
 *   W      Work space for blocking invocations, matrix of size N-by-nb.
 *
 *   ipiv   Pivot indeces, for each non-zero element ipiv[k] the k'th row is exchanged with
 *          ipiv[k]-1'th row.
 *
 *   flags  Indicator bits. 
 *
 *   nb     Blocking factor for blocked invocations. If bn == 0 or
 *          N < nb unblocked algorithm is used.
 *
 * Returns:
 *  LDL factorization and error indicator.
 *
 */
func DecomposeLDL(A, W *matrix.FloatMatrix, ipiv []int, flags Flags, nb int) (*matrix.FloatMatrix, error) {
    var err error
    if A.Cols() != A.Rows() {
        return nil, errors.New("A not a square matrix")
    }
    for k, _ := range ipiv {
        ipiv[k] = 0
    }
    if A.Cols() < nb || nb == 0 {
        if flags & LOWER != 0 {
            err = unblkLowerLDL(A, &pPivots{ipiv})
        } else {
            err = unblkUpperLDL(A, &pPivots{ipiv})
        }
    } else {
        if flags & LOWER != 0 {
            err = blkLowerLDL(A, W, &pPivots{ipiv}, nb)
        } else {
            err = blkUpperLDL(A, W, &pPivots{ipiv}, nb)
        }
    }
    return A, err
}

/*
 * Solves a system system of linear equations A*X = B with symmetric positive
 * definite matrix A using the LDL factorization A = L*D*L.T or A = U*D*U.T
 * computed by DecomposeLDL().
 *
 * Arguments:
 *  B      On entry, the right hand side matrix B. On exit, the solution
 *         matrix X.
 *
 *  A      The triangular factor U or L from LDL factorization as computed by
 *         DecomposeLDL().
 *
 *  ipiv   Pivot indeces, for each non-zero element ipiv[k] the k'th row is exchanged with
 *         ipiv[k]-1'th row.
 *
 *  flags  Indicator of which factor is stored in A. If flags&UPPER then upper
 *         triangle of A is stored. If flags&LOWER then lower triangle of A is
 *         stored.
 */
func SolveLDL(B, A *matrix.FloatMatrix, ipiv []int, flags Flags)  {
    if flags & UPPER != 0 {
        // X = (U*D*U.T).-1*B => U.-T*(D.-1*(U.-1*B))
        SolveTrm(B, A, 1.0, UPPER|UNIT)
        SolveDiag(B, A, LEFT)
        SolveTrm(B, A, 1.0, UPPER|UNIT|TRANSA)

    } else if flags & LOWER != 0 {
        // X = (L*D*L.T).-1*B = L.-T*(D*-1(L.-1*B))
        // arrange to match factorization
        applyRowPivots(B, &pPivots{ipiv}, 0, FORWARD)
        // solve
        SolveTrm(B, A, 1.0, LOWER|UNIT)
        SolveDiag(B, A, LEFT)
        SolveTrm(B, A, 1.0, LOWER|UNIT|TRANSA)
        // rearrange to original
        applyRowPivots(B, &pPivots{ipiv}, 0, FORWARD)
    }
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
