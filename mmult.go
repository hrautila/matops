
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    "github.com/hrautila/matops/calgo"
    "errors"
    "math"
    //"fmt"
)

// blocking parameter size for AXPY based algorithms
var vpLenAxpy int = 48
var nBaxpy int = 256
var mBaxpy int = 256

// blocking parameter size for DOT based algorithms
var vpLenDot int = 196
var nBdot int = 68
var mBdot int = 68

// Number of parallel workers to use.
var nWorker int = 1

// problems small than this do not benefit from parallelism
var limitOne int = 200*200

func ParamsDot(vplen, nb, mb int) {
    vpLenDot = vplen
    nBdot = nb
    mBdot = mb
}

func NumWorkers(newWorkers int) int {
    oldWorkers := nWorker
    nWorker = newWorkers
    return oldWorkers
}

func blockIndex4(i, r, sz int) int {
    if (i == r) {
        return sz;
    }
    return i*sz/r - ((i*sz/r) & 0x3);
}

func blockIndex2(i, r, sz int) int {
    if (i == r) {
        return sz;
    }
    return i*sz/r - ((i*sz/r) & 0x1);
}

func isSquared(num int) (int, bool) {
    nsqrt := int(math.Sqrt(float64(num)))
    issquared := nsqrt*nsqrt == num
    return nsqrt, issquared
}


func divideWork(rows, cols, workers int) (colWorkers int, rowWorkers int) {
    colWorkers = 0
    rowWorkers = 0
    nwsqrt := int(math.Sqrt(float64(workers)))
    issquare := nwsqrt*nwsqrt == workers
    if workers == 2 || (workers & 0x1) != 0 {
        // odd number of workers
        if cols > rows {
            colWorkers = workers
            rowWorkers = 1
        } else {
            rowWorkers = workers
            colWorkers = 1
        }
    } else if issquare {
        // square number 
        colWorkers = nwsqrt
        rowWorkers = nwsqrt
    } else {
        // even number of workers
        if cols > rows {
            rowWorkers = 2
            colWorkers = workers/2
        } else {
            colWorkers = 2
            rowWorkers = workers/2
        }
    }    
    //fmt.Printf("divideWork: c=%d, r=%d\n", colWorkers, rowWorkers)
    return
}

type task func(int, int, int, int, chan int)

func scheduleWork(colworks, rowworks, cols, rows int, worker task) {
    ntask := colworks*rowworks
    ch := make(chan int, ntask)
    for k := 0; k < colworks; k++ {
        colstart := blockIndex4(k, colworks, cols)
        colend   := blockIndex4(k+1, colworks, cols)
        for l := 0; l < rowworks; l++ {
            rowstart := blockIndex4(l, rowworks, rows)
            rowend   := blockIndex4(l+1, rowworks, rows)
            //fmt.Printf("schedule: S=%d, L=%d, R=%d, E=%d\n", colstart, colend, rowstart, rowend)
            go worker(colstart, colend, rowstart, rowend, ch)
        }
    }
    nready := 0
    for nready < ntask {
        nready += <- ch
    }
}

// Calculate C = alpha*A*B + beta*C, C is M*N, A is M*P and B is P*N
func MMMult(C, A, B *matrix.FloatMatrix, alpha, beta float64) error {
    psize := C.NumElements()
    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Br := B.FloatArray()
    ldB := B.LeadingIndex()
    Cr := C.FloatArray()
    ldC := C.LeadingIndex()

    if nWorker <= 1 || psize <= limitOne {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.NOTRANS, ldC, ldA, ldB, B.Rows(),
            0, C.Cols(), 0, C.Rows(),
            vpLenDot, nBdot, mBdot)
        return nil
    } 
    // here we have more than one worker available
    worker := func(cstart, cend, rstart, rend int, ready chan int) {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.NOTRANS, ldC, ldA, ldB, B.Rows(),
            cstart, cend, rstart, rend, vpLenDot, nBdot, mBdot)
        ready <- 1
    }
    colworks, rowworks := divideWork(C.Rows(), C.Cols(), nWorker)
    scheduleWork(colworks, rowworks, C.Cols(), C.Rows(), worker)
    return nil
}

// Calculate C = alpha*A.T*B + beta*C, C is M*N, A is P*M and B is P*N
func MMMultTransA(C, A, B *matrix.FloatMatrix, alpha, beta float64) error {
    psize := C.NumElements()
    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Br := B.FloatArray()
    ldB := B.LeadingIndex()
    Cr := C.FloatArray()
    ldC := C.LeadingIndex()
    if nWorker <= 1 || psize <= limitOne {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.TRANSA, ldC, ldA, ldB,
            B.Rows(), 0, C.Cols(), 0, C.Rows(), vpLenDot, nBdot, mBdot)
        return nil
    }

    // here we have more than one worker available
    worker := func(cstart, cend, rstart, rend int, ready chan int) {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.TRANSA, ldC, ldA, ldB, B.Rows(),
            cstart, cend, rstart, rend, vpLenDot, nBdot, mBdot)
        ready <- 1
    }
    colworks, rowworks := divideWork(C.Rows(), C.Cols(), nWorker)
    scheduleWork(colworks, rowworks, C.Cols(), C.Rows(), worker)
    //scheduleWork(colworks, rowworks, worker)
    return nil
}

// Calculate C = alpha*A*B.T + beta*C, C is M*N, A is M*P and B is N*P
func MMMultTransB(C, A, B *matrix.FloatMatrix, alpha, beta float64) error {
    psize := C.NumElements()
    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Br := B.FloatArray()
    ldB := B.LeadingIndex()
    Cr := C.FloatArray()
    ldC := C.LeadingIndex()
    if nWorker <= 1 || psize <= limitOne {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.TRANSB, ldC, ldA, ldB,
            B.Rows(), 0, C.Cols(), 0, C.Rows(), vpLenDot, nBdot, mBdot)
        return nil
    }

    // here we have more than one worker available
    worker := func(cstart, cend, rstart, rend int, ready chan int) {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.TRANSB, ldC, ldA, ldB, B.Rows(),
            cstart, cend, rstart, rend, vpLenDot, nBdot, mBdot)
        ready <- 1
    }
    colworks, rowworks := divideWork(C.Rows(), C.Cols(), nWorker)
    scheduleWork(colworks, rowworks, C.Cols(), C.Rows(), worker)
    //scheduleWork(colworks, rowworks, worker)
    return nil
}

// Calculate C = alpha*A.T*B.T + beta*C, C is M*N, A is P*M and B is N*P
func MMMultTransAB(C, A, B *matrix.FloatMatrix, alpha, beta float64) error {
    psize := C.NumElements()
    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Br := B.FloatArray()
    ldB := B.LeadingIndex()
    Cr := C.FloatArray()
    ldC := C.LeadingIndex()
    if nWorker <= 1 || psize <= limitOne{
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.TRANSA|calgo.TRANSB, ldC, ldA, ldB,
            B.Rows(), 0, C.Cols(), 0, C.Rows(), vpLenDot, nBdot, mBdot)
        return nil
    }

    // here we have more than one worker available
    worker := func(cstart, cend, rstart, rend int, ready chan int) {
        calgo.DMult(Cr, Ar, Br, alpha, beta, calgo.TRANSA|calgo.TRANSB, ldC, ldA, ldB,
            B.Rows(), cstart, cend, rstart, rend, vpLenDot, nBdot, mBdot)
        ready <- 1
    }
    colworks, rowworks := divideWork(C.Rows(), C.Cols(), nWorker)
    scheduleWork(colworks, rowworks, C.Cols(), C.Rows(), worker)
    //scheduleWork(colworks, rowworks, worker)
    return nil
}


// Calculate C = alpha*A*B + beta*C, C is M*N, A is M*M and B is M*N
func MMSymm(C, A, B *matrix.FloatMatrix, alpha, beta float64) error {

    if A.Rows() != A.Cols() {
        return errors.New("A matrix not square matrix.");
    }
    psize := C.NumElements()
    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Br := B.FloatArray()
    ldB := B.LeadingIndex()
    Cr := C.FloatArray()
    ldC := C.LeadingIndex()

    if nWorker <= 1 || psize <= limitOne {
        calgo.DMultSymm(Cr, Ar, Br, alpha, beta, calgo.LEFT|calgo.LOWER, ldC, ldA, ldB,
            A.Cols(),  0, C.Cols(), 0, C.Rows(), vpLenDot, nBdot, mBdot)
        return nil
    } 
    // here we have more than one worker available
    worker := func(cstart, cend, rstart, rend int, ready chan int) {
        calgo.DMultSymm(Cr, Ar, Br, alpha, beta, calgo.LEFT|calgo.LOWER, ldC, ldA, ldB,
            A.Cols(), cstart, cend, rstart, rend, vpLenDot, nBdot, mBdot)
        ready <- 1
    }
    colworks, rowworks := divideWork(C.Rows(), C.Cols(), nWorker)
    scheduleWork(colworks, rowworks, C.Cols(), C.Rows(), worker)
    return nil
}

// Calculate C = alpha*A*B + beta*C, C is M*N, A is M*M and B is M*N
func MMSymmUpper(C, A, B *matrix.FloatMatrix, alpha, beta float64) error {

    if A.Rows() != A.Cols() {
        return errors.New("A matrix not square matrix.");
    }
    psize := C.NumElements()
    Ar := A.FloatArray()
    ldA := A.LeadingIndex()
    Br := B.FloatArray()
    ldB := B.LeadingIndex()
    Cr := C.FloatArray()
    ldC := C.LeadingIndex()

    if nWorker <= 1 || psize <= limitOne {
        calgo.DMultSymm(Cr, Ar, Br, alpha, beta, calgo.LEFT|calgo.UPPER, ldC, ldA, ldB,
            A.Cols(), 0, C.Cols(), 0, C.Rows(), vpLenDot, nBdot, mBdot)
        return nil
    } 
    // here we have more than one worker available
    worker := func(cstart, cend, rstart, rend int, ready chan int) {
        calgo.DMultSymm(Cr, Ar, Br, alpha, beta, calgo.LEFT|calgo.UPPER, ldC, ldA, ldB,
            A.Cols(), cstart, cend, rstart, rend, vpLenDot, nBdot, mBdot)
        ready <- 1
    }
    colworks, rowworks := divideWork(C.Rows(), C.Cols(), nWorker)
    scheduleWork(colworks, rowworks, C.Cols(), C.Rows(), worker)
    return nil
}

// Y = alpha*A*X + beta*Y
func MVMult(Y, A, X *matrix.FloatMatrix, alpha, beta float64) error {

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
        0, lenX, 0, lenY, vpLenDot, mBdot)
    return nil
}

// Y = alpha*A.T*X + beta*Y
func MVMultTransA(Y, A, X *matrix.FloatMatrix, alpha, beta float64) error {

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
    calgo.DMultMV(Yr, Ar, Xr, alpha, beta, calgo.TRANSA, incY, ldA, incX,
        0, lenX, 0, lenY, vpLenDot, mBdot)
    return nil
}

// A = A + alpha*X*Y.T; A is M*N, X is row or column vector of length M and
// Y is row or column vector of legth N.
func MVRankUpdate(A, X, Y *matrix.FloatMatrix, alpha float64) error {

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
    calgo.DRankMV(Ar, Xr, Yr, alpha, ldA, incY, incX, 0, A.Cols(), 0, A.Rows(), 0, 0)
    return nil
}

// A = A + alpha*X*Y.T; A is N*N symmetric, X is row or column vector of length N.
func MVSymmUpdate(A, X *matrix.FloatMatrix, alpha float64) error {

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
    calgo.DSymmRankMV(Ar, Xr, alpha, calgo.LOWER, ldA, incX, 0, A.Cols(), 0)
    return nil
}

// A = A + alpha*X*Y.T; A is N*N symmetric, X is row or column vector of length N.
func MVSymmUpdateUpper(A, X *matrix.FloatMatrix, alpha float64) error {

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
    calgo.DSymmRankMV(Ar, Xr, alpha, calgo.UPPER, ldA, incX, 0, A.Cols(), 0)
    return nil
}

// A = A + alpha*X*Y.T; A is M*N, X is row or column vector of length M and
// Y is row or column vector of legth N.
func MVSymm2Update(A, X, Y *matrix.FloatMatrix, alpha float64) error {

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
    calgo.DSymmRank2MV(Ar, Xr, Yr, alpha, calgo.LOWER, ldA, incY, incX, 0, A.Cols(), 0)
    return nil
}

func MVSymm2UpdateUpper(A, X, Y *matrix.FloatMatrix, alpha float64) error {

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
    calgo.DSymmRank2MV(Ar, Xr, Yr, alpha, calgo.UPPER, ldA, incY, incX, 0, A.Cols(), 0)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
