
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "github.com/hrautila/matops/calgo"
    "errors"
)

// Y = alpha*A*X + beta*Y
func MVMult(Y, A, X *matrix.FloatMatrix, alpha, beta float64) error {

    if A.Rows() == 0 || A.Cols() == 0 {
        return nil
    }
    if Y.Rows() != 1 && Y.Cols() != 1 {
        return errors.New("Y not a vector.");
    }
    if X.Rows() != 1 && X.Cols() != 1 {
        return errors.New("X not a vector.");
    }

    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Yr := Y.FloatArray()
    incY := 1
    lenY := Y.Rows()
    if Y.Rows() == 1 {
        // row vector
        incY = Y.LeadingIndex()
        lenY = Y.Cols()
    }
    Xr := X.FloatArray()
    incX := 1
    lenX := X.Rows()
    if X.Rows() == 1 {
        // row vector
        incX = X.LeadingIndex()
        lenX = X.Cols()
    }
    // NOTE: This could diveded to parallel tasks by rows.
    calgo.DMultMV(Yr, Ar, Xr, alpha, beta, calgo.NULL, incY, ldA, incX,
        0, lenX, 0, lenY, vpLen, mB)
    return nil
}


// Matrix-vector rank update A = A + alpha*X*Y.T
//    A is M*N generic matrix,
//    X is row or column vector of length M 
//    Y is row or column vector of legth N.
func MVRankUpdate(A, X, Y *matrix.FloatMatrix, alpha float64) error {

    if A.Rows() == 0 || A.Cols() == 0 {
        return nil
    }
    if Y.Rows() != 1 && Y.Cols() != 1 {
        return errors.New("Y not a vector.");
    }
    if X.Rows() != 1 && X.Cols() != 1 {
        return errors.New("X not a vector.");
    }

    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Yr := Y.FloatArray()
    incY := 1
    if Y.Rows() == 1 {
        // row vector
        incY = Y.LeadingIndex()
    }
    Xr := X.FloatArray()
    incX := 1
    if X.Rows() == 1 {
        // row vector
        incX = X.LeadingIndex()
    }
    // NOTE: This could diveded to parallel tasks like matrix-matrix multiplication
    calgo.DRankMV(Ar, Xr, Yr, alpha, ldA, incX, incY, 0, A.Cols(), 0, A.Rows(), 0, 0)
    return nil
}

// Matrix-vector symmetric rank update A = A + alpha*X*X.T
//   A is N*N symmetric,
//   X is row or column vector of length N.
func MVRankUpdateSym(A, X *matrix.FloatMatrix, alpha float64, flags Flags) error {

    if A.Rows() == 0 || A.Cols() == 0 {
        return nil
    }
    if X.Rows() != 1 && X.Cols() != 1 {
        return errors.New("X not a vector.");
    }

    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Xr := X.FloatArray()
    incX := 1
    if X.Rows() == 1 {
        // row vector
        incX = X.LeadingIndex()
    }
    // NOTE: This could diveded to parallel tasks per column.
    calgo.DSymmRankMV(Ar, Xr, alpha, calgo.Flags(flags), ldA, incX, 0, A.Cols(), 0)
    return nil
}

// Matrix-vector symmetric rank 2 update A = A + alpha*X*Y.T + alpha*X.T*Y
//   A is N*N symmetric matrix,
//   X is row or column vector of length N
//   Y is row or column vector of legth N.
func MVRankUpdate2Sym(A, X, Y *matrix.FloatMatrix, alpha float64, flags Flags) error {

    if A.Rows() == 0 || A.Cols() == 0 {
        return nil
    }
    if Y.Rows() != 1 && Y.Cols() != 1 {
        return errors.New("Y not a vector.");
    }
    if X.Rows() != 1 && X.Cols() != 1 {
        return errors.New("X not a vector.");
    }

    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Yr := Y.FloatArray()
    incY := 1
    if Y.Rows() == 1 {
        // row vector
        incY = Y.LeadingIndex()
    }
    Xr := X.FloatArray()
    incX := 1
    if X.Rows() == 1 {
        // row vector
        incX = X.LeadingIndex()
    }
    // NOTE: This could diveded to parallel tasks like matrix-matrix multiplication
    calgo.DSymmRank2MV(Ar, Xr, Yr, alpha, calgo.Flags(flags), ldA, incY, incX, 0, A.Cols(), 0)
    return nil
}


// Matrix-vector solve X = A.-1*X or X = A.-T*X
//   A is N*N tridiagonal lower or upper,
//   X is row or column vector of length N.
// flags
//   LOWER  A is lower tridiagonal
//   UPPER  A is upper tridiagonal
//   UNIT   A diagonal is unit
//   TRANSA A is transpose
func MVSolve(X, A *matrix.FloatMatrix, alpha float64, flags Flags) error {

    if A.Rows() == 0 || A.Cols() == 0 {
        return nil
    }
    if X.Rows() != 1 && X.Cols() != 1 {
        return errors.New("X not a vector.");
    }

    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Xr := X.FloatArray()
    incX := 1
    if X.Rows() == 1 {
        // row vector
        incX = X.LeadingIndex()
    }
    calgo.DSolveBlkMV(Xr, Ar, calgo.Flags(flags), incX, ldA, A.Cols(), nB)
    return nil
}


// Tridiagonal multiplication; X = A*X
func MVMultTrm(X, A*matrix.FloatMatrix, flags Flags) error {

    if A.Rows() == 0 || A.Cols() == 0 {
        return nil
    }
    if X.Rows() != 1 && X.Cols() != 1 {
        return errors.New("X not a vector.");
    }

    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Xr := X.FloatArray()
    incX := 1
    if X.Rows() == 1 {
        // row vector
        incX = X.LeadingIndex()
    }
    calgo.DTrimvUnblkMV(Xr, Ar, calgo.Flags(flags), incX, ldA, A.Cols())
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
