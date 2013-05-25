
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


/*
 * Unblocked algorith for computing C = Q.T*C and C = Q*C.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th column
 * below diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces H(k)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(k) == Q and C = Q*C
 */
func unblockedMultQLeft(C, A, tau, w *matrix.FloatMatrix, flags Flags) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a10, a11, A20, a21, A22 matrix.FloatMatrix
    var CT, CB, C0, c1t, C2 matrix.FloatMatrix
    var tT, tB matrix.FloatMatrix
    var t0, tau1, t2  matrix.FloatMatrix
    var Aref *matrix.FloatMatrix
    var pAdir, pAstart, pDir, pStart pDirection
    var mb int

    // partitioning start and direction
    if flags & TRANS != 0 {
        pAstart = pTOPLEFT
        pAdir   = pBOTTOMRIGHT
        pStart  = pTOP
        pDir    = pBOTTOM
        mb = 0
        Aref = &ABR
    } else {
        pAstart = pBOTTOMRIGHT
        pAdir   = pTOPLEFT
        pStart  = pBOTTOM
        pDir    = pTOP
        mb    = C.Rows() - C.Cols()
        Aref = &ATL
    }

    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,  A, mb, 0, pAstart)
    partition2x1(
        &CT,
        &CB,    C, mb, pStart)
    partition2x1(
        &tT,
        &tB,  tau, 0, pStart)

    for Aref.Rows() > 0 && Aref.Cols() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, pAdir)
        repartition2x1to3x1(&CT,
            &C0,
            &c1t,
            &C2,     C, 1, pDir)
        repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, pDir)

        // --------------------------------------------------------

        applyHHTo2x1(&tau1, &a21, &c1t, &C2, w, LEFT)

        // --------------------------------------------------------
        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, pAdir)
        continue3x1to2x1(
            &CT,
            &CB,   &C0, &c1t,   C, pDir)
        continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   tau, pDir)
    }
}

func unblockedMultQRight(C, A, tau, w *matrix.FloatMatrix, flags Flags) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, a10, a11, A20, a21, A22 matrix.FloatMatrix
    var CL, CR, C0, c1, C2 matrix.FloatMatrix
    var tT, tB matrix.FloatMatrix
    var t0, tau1, t2  matrix.FloatMatrix
    var Aref *matrix.FloatMatrix
    var pAdir, pAstart, pDir, pStart, pCstart, pCdir pDirection
    var mb int

    // partitioning start and direction
    if flags & TRANS != 0 {
        pAstart = pTOPLEFT
        pAdir   = pBOTTOMRIGHT
        pStart  = pTOP
        pDir    = pBOTTOM
        pCstart = pLEFT
        pCdir   = pRIGHT
        mb = 0
        Aref = &ABR
    } else {
        pAstart = pBOTTOMRIGHT
        pAdir   = pTOPLEFT
        pStart  = pBOTTOM
        pDir    = pTOP
        pCstart = pRIGHT
        pCdir   = pLEFT
        mb    = C.Rows() - C.Cols()
        Aref = &ATL
    }

    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,  A, mb, 0, pAstart)
    partition1x2(
        &CL, &CR,    C, 0, pCstart)
    partition2x1(
        &tT,
        &tB,  tau, 0, pStart)

    for Aref.Rows() > 0 && Aref.Cols() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, pAdir)
        repartition1x2to1x3(&CL,
            &C0, &c1, &C2,      C, 1, pCdir)
        repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, pDir)

        // --------------------------------------------------------

        applyHHTo2x1(&tau1, &a21, &c1, &C2, w, RIGHT)

        // --------------------------------------------------------
        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, pAdir)
        continue3x1to2x1(
            &CL, &CR,   &C0, &c1,   C, pCdir)
        continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   tau, pDir)
    }
}


/*
 * Blocked version for computing C = Q*C and C = Q.T*C from elementary reflectors
 * and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block reflector T.
 * Matrix C is updated by applying block reflector T using compact WY algorithm.
 */
func blockedMultQLeft(C, A, tau, W *matrix.FloatMatrix, nb int, flags Flags) {
    var ATL, ATR, ABL, ABR, AL, AR matrix.FloatMatrix
    var A00, A10, A11, A20, A21, A22 matrix.FloatMatrix
    var CT, CB, C0, C1, C2 matrix.FloatMatrix
    var tT, tB matrix.FloatMatrix
    var t0, tau1, t2  matrix.FloatMatrix
    var Wrk matrix.FloatMatrix

    Twork := matrix.FloatZeros(nb, nb)

    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, pTOPLEFT)
    partition2x1(
        &CT,
        &CB,  C, 0, pTOP)
    partition2x1(
        &tT,
        &tB,  tau, 0, pTOP)

    transpose := flags & TRANS != 0

    for ABR.Rows() > 0 && ABR.Cols() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pBOTTOMRIGHT)
        repartition2x1to3x1(&CT,
            &C0,
            &C1,
            &C2,     C, nb, pBOTTOM)
        repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, nb, pBOTTOM)

        // --------------------------------------------------------
        // divide bottom right block to left and right
        partition1x2(
            &AL, &AR, &ABR, nb, pLEFT)

        // build block reflector from left block
        unblkQRBlockReflector(Twork, &AL, &tau1)

        // compute: Q*T.C == C - Y*(C.T*Y*T).T  transpose == true
        //          Q*C   == C - C*Y*T*Y.T      transpose == false
        Wrk.SubMatrixOf(W, 0, 0, C1.Cols(), nb)
        updateWithQT(&C1, &C2, &A11, &A21, Twork, &Wrk, nb, transpose)

        // --------------------------------------------------------
        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pBOTTOMRIGHT)
        continue3x1to2x1(
            &CT,
            &CB,   &C0, &C1,   C, pBOTTOM)
        continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   tau, pBOTTOM)
    }

}

/*
 * Blocked version for computing C = Q*C and C = Q.T*C with block reflector.
 *
 */
func blockedMultQTLeft(C, A, T, W *matrix.FloatMatrix, nb int, flags Flags) {
    var ATL, ATR, ABL, ABR matrix.FloatMatrix
    var A00, A10, A11, A20, A21, A22 matrix.FloatMatrix
    var CT, CB, C0, C1, C2 matrix.FloatMatrix
    var TTL, TTR, TBL, TBR matrix.FloatMatrix
    var T00, T01, T02, T11, T12, T22 matrix.FloatMatrix

    partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, pTOPLEFT)
    partition2x2(
        &TTL, &TTR,
        &TBL, &TBR,   T, 0, 0, pTOPLEFT)
    partition2x1(
        &CT,
        &CB,  C, 0, pTOP)

    transpose := flags & TRANS != 0

    for ABR.Rows() > 0 && ABR.Cols() > 0 {
        repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pBOTTOMRIGHT)
        repartition2x2to3x3(&TTL,
            &T00, &T01, &T02,
            nil,  &T11, &T12,
            nil,  nil,  &T22,   T, nb, pBOTTOMRIGHT)
        repartition2x1to3x1(&CT,
            &C0,
            &C1,
            &C2,     C, nb, pBOTTOM)

        // --------------------------------------------------------

        // compute: Q*T.C == C - Y*(C.T*Y*T).T  transpose == true
        //          Q*C   == C - C*Y*T*Y.T      transpose == false

        var Wrk matrix.FloatMatrix
        Wrk.SubMatrixOf(W, 0, 0, C1.Cols(), T11.Cols())
        updateWithQT(&C1, &C2, &A11, &A21, &T11, &Wrk, nb, transpose)

        // --------------------------------------------------------
        continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pBOTTOMRIGHT)
        continue3x3to2x2(
            &TTL, &TTR,
            &TBL, &TBR,   &T00, &T11, &T22,   T, pBOTTOMRIGHT)
        continue3x1to2x1(
            &CT,
            &CB,   &C0, &C1,   C, pBOTTOM)
    }

}

/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 * as returned by DecomposeQR().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C. On exit C is overwritten by Q*C or Q.T*C.
 *
 *  A     QR factorization as returne by DecomposeQR() where the lower trapezoidal
 *        part holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors.
 *
 *  W     Workspace, used for blocked invocations. Size C.Cols()-by-nb.
 *
 *  nb    Blocksize for blocked invocations. If C.Cols() <= nb unblocked algorithm
 *        is used.
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS, NOTRANS
 *       
 * Compatible with lapack.DORMRQ
 */
func MultQ(C, A, tau, W *matrix.FloatMatrix, flags Flags, nb int) error {
    var err error = nil
    if flags & RIGHT != 0 {
        if C.Cols() != A.Rows() {
            return errors.New("C*Q: C.Cols != A.Rows")
        }
    } else {
        // default is from LEFT
        if C.Rows() != A.Rows() {
            return errors.New("Q*C: C.Rows != A.Rows")
        }
    }
    if nb == 0 || C.Cols() < nb {
        w := matrix.FloatZeros(1, C.Cols())
        unblockedMultQLeft(C, A, tau, w, flags)
    } else {
        if W == nil {
            return errors.New("workspace not defined")
        } else if W.Cols() < nb || W.Rows() < C.Cols() {
            return errors.New("workspace too small")
        }
        var Wrk matrix.FloatMatrix
        Wrk.SubMatrixOf(W, 0, 0, C.Cols(), nb)
        blockedMultQLeft(C, A, tau, &Wrk, nb, flags)
    }
    return err
}


/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors and block reflector T
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 * as returned by DecomposeQRT().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C. On exit C is overwritten by Q*C or Q.T*C.
 *
 *  A     QR factorization as returne by DecomposeQRT() where the lower trapezoidal
 *        part holds the elementary reflectors.
 *
 *  T     The block reflector computed from elementary reflectors as returned by
 *        DecomposeQRT() or computed from elementary reflectors and scalar coefficients
 *        BuildT()
 *
 *  W     Workspace, size C.Cols()-by-nb or C.Rows()-by-nb
 *
 *  nb    Blocksize for blocked invocations. If nb == 0 default value T.Cols() 
 *        is used.
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS, NOTRANS
 *       
 * Compatible with lapack.DORMRQ
 */
func MultQT(C, A, T, W *matrix.FloatMatrix, flags Flags, nb int) error {
    var err error = nil
    if flags & RIGHT != 0 {
        if C.Cols() != A.Rows() {
            return errors.New("C*Q: C.Cols != A.Rows")
        }
    } else {
        // default is from LEFT
        if C.Rows() != A.Rows() {
            return errors.New("Q*C: C.Rows != A.Rows")
        }
    }
    if nb == 0  {
        nb = T.Cols()
    }
    if W == nil {
        return errors.New("workspace not defined")
    } else if W.Cols() < nb || W.Rows() < C.Cols() {
        return errors.New("workspace too small")
    }

    var Wrk matrix.FloatMatrix
    Wrk.SubMatrixOf(W, 0, 0, C.Cols(), nb)
    blockedMultQTLeft(C, A, T, &Wrk, nb, flags)

    return err
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
