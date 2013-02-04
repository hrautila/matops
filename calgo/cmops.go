
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


package calgo

// #cgo CFLAGS: -O3 -msse2 -funroll-loops -fomit-frame-pointer -ffast-math 
// #include "cmops.h"
import "C"
import "unsafe"


// C = alpha*A*B + beta*C; C is M*N, A is M*P and B is P*N; all data aligned to 16 bytesf
func MultAligned(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {

    var Cm C.mdata_t
    var Am C.mdata_t
    var Bm C.mdata_t

    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_aligned_notrans(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// C = alpha*A*B + beta*C; C is M*N, A is M*P and B is P*N; data may be unaligned
func MultUnAligned(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_unaligned_notrans(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}

// C = alpha*A.T*B + beta*C; C is M*N, A is P*M and B is P*N; all data aligned to 16 bytes
func MultAlignedTransA(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_aligned_transa(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// C = alpha*A.T*B + beta*C; C is M*N, A is P*M and B is P*N; data may be unaligned
func MultUnAlignedTransA(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_unaligned_transa(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// C = alpha*A*B.T + beta*C; C is M*N, A is M*P and B is N*P; all data aligned to 16 bytes
func MultAlignedTransB(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_aligned_transb(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// C = alpha*A*B.T + beta*C; C is M*N, A is M*P and B is N*P; data may be unaligned
func MultUnAlignedTransB(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_unaligned_transb(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// C = alpha*A.T*B.T + beta*C; C is M*N, A is P*M and B is N*P;
// all data aligned to 16 bytes
func MultAlignedTransAB(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_aligned_transab(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// C = alpha*A.T*B.T + beta*C; C is M*N, A is P*M and B is N*P; data may be unaligned
func MultUnAlignedTransAB(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, P, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_unaligned_transab(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}

// C = alpha*A*B + beta*C; C is N*P, A is N*N and B is N*P; data may be unaligned
func MultSymmUnAligned(C, A, B []float64, alpha, beta float64, ldC, ldA, ldB, N, S, L, R, E, H, NB, MB int) {
    var Cm C.mdata_t;
    var Am C.mdata_t;
    var Bm C.mdata_t;
    Cm.md =  (*C.double)(unsafe.Pointer(&C[0]))
    Cm.step = C.int(ldC)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)
    Bm.md =  (*C.double)(unsafe.Pointer(&B[0]))
    Bm.step = C.int(ldB)

    C.dmult_symm_ua_notrans(
        (*C.mdata_t)(unsafe.Pointer(&Cm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta),
        C.int(N), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(NB), C.int(MB))
}


// Y = alpha*A*X + beta*Y; Y is M*1, X is N*1 and A is M*N
func MultMV(Y, A, X []float64, alpha, beta float64, incY, ldA, incX, S, L, R, E, H, MB int) {
    var Yv C.mvec_t;
    var Xv C.mvec_t;
    var Am C.mdata_t;
    Yv.md =  (*C.double)(unsafe.Pointer(&Y[0]))
    Yv.inc = C.int(incY)
    Xv.md =  (*C.double)(unsafe.Pointer(&X[0]))
    Xv.inc = C.int(incX)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)

    C.dmult_mv_notrans(
        (*C.mvec_t)(unsafe.Pointer(&Yv)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mvec_t)(unsafe.Pointer(&Xv)),
        C.double(alpha), C.double(beta),
        C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(MB))
}

// Y = alpha*A*X + beta*Y; Y is M*1, X is N*1 and A is N*M
func MultMVTransA(Y, A, X []float64, alpha, beta float64, incY, ldA, incX, S, L, R, E, H, MB int) {
    var Yv C.mvec_t;
    var Xv C.mvec_t;
    var Am C.mdata_t;
    Yv.md =  (*C.double)(unsafe.Pointer(&Y[0]))
    Yv.inc = C.int(incY)
    Xv.md =  (*C.double)(unsafe.Pointer(&X[0]))
    Xv.inc = C.int(incX)
    Am.md =  (*C.double)(unsafe.Pointer(&A[0]))
    Am.step = C.int(ldA)

    C.dmult_mv_transa(
        (*C.mvec_t)(unsafe.Pointer(&Yv)),
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        (*C.mvec_t)(unsafe.Pointer(&Xv)),
        C.double(alpha), C.double(beta),
        C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(H), C.int(MB))
}



func copy_trans(C, A []float64, ldC, ldA, M, N int) {
    var Cr *C.double
    var Ar *C.double
    Cr =  (*C.double)(unsafe.Pointer(&C[0]))
    Ar =  (*C.double)(unsafe.Pointer(&A[0]))
    C.colcpy_trans(Cr, C.int(ldC), Ar, C.int(ldA), C.int(M), C.int(N))
}




// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

