
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package matops

import (
    "github.com/hrautila/matrix"
    //"fmt"
)

// Functions here are support functions for libFLAME-like implementation
// of various linear algebra algorithms.

/*
 Partition A to 2 by 2 blocks.

           ATL | ATR
  A  -->   =========
           ABL | ABR

 Parameter nb is initial block size for ATL. 
 */
func partition2x2(ATL, ATR, ABL, ABR, A *matrix.FloatMatrix, nb int) {
    ATL.SubMatrixOf(A, 0, 0,  nb, nb)
    ATR.SubMatrixOf(A, 0, nb, nb, A.Cols()-nb)
    ABL.SubMatrixOf(A, nb, 0, A.Rows()-nb, nb)
    ABR.SubMatrixOf(A, nb, nb)
}

/*
 Repartition 2 by 2 blocks to 3 by 3 blocks.

                      A00 | A01 : A02
   ATL | ATR   nb     ===============
   =========   -->    A10 | A11 : A12
   ABL | ABR          ---------------
                      A20 | A21 : A22

   ATR, ABL, ABR implicitely defined by ATL and A.
 */
func repartition2x2to3x3(ATL, 
    A00, A01, A02, A10, A11, A12, A20, A21, A22, A *matrix.FloatMatrix, nb int) {

    k := ATL.Rows()
    if k + nb > A.Cols() {
        nb = A.Cols() - k
    }
    A00.SubMatrixOf(A, 0, 0,    k, k)
    A01.SubMatrixOf(A, 0, k,    k, nb)
    A02.SubMatrixOf(A, 0, k+nb, k, A.Cols()-k-nb)

    A10.SubMatrixOf(A, k, 0,    nb, k)
    A11.SubMatrixOf(A, k, k,    nb, nb)
    A12.SubMatrixOf(A, k, k+nb, nb, A.Cols()-k-nb)

    A20.SubMatrixOf(A, k+nb, 0,    nb, k)
    A21.SubMatrixOf(A, k+nb, k,    nb, nb)
    A22.SubMatrixOf(A, k+nb, k+nb)
}

/*
 Redefine 2 by 2 blocks from 3 by 3 partition.

                      A00 : A01 | A02
   ATL | ATR   nb     ---------------
   =========   <--    A10 : A11 | A12
   ABL | ABR          ===============
                      A20 : A21 | A22

 New division of ATL, ATR, ABL, ABR defined by diagonal entries A00, A11, A22
 */
func continue3x3to2x2(
    ATL, ATR, ABL, ABR, 
    A00, A11, A22, A *matrix.FloatMatrix) {

    k := A00.Rows()
    mb := A11.Cols()
    ATL.SubMatrixOf(A, 0, 0,    k+mb, k+mb)
    ATR.SubMatrixOf(A, 0, k+mb, k+mb, A.Cols()-k-mb)

    ABL.SubMatrixOf(A, k+mb, 0, A.Rows()-k-mb, k+mb)
    ABR.SubMatrixOf(A, k+mb, k+mb)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
