
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

type pDirection int
const (
    pLEFT = iota
    pRIGHT 
    pTOP
    pBOTTOM
    pTOPLEFT
    pBOTTOMRIGHT
)

/*
 Partition p to 2 by 1 blocks.

        AT
  A --> --
        AB

 Parameter nb is initial block size for AT (pTOP) or AB (pBOTTOM).  
 */
func partition2x1(AT, AB, A *matrix.FloatMatrix, nb, side int) {
    switch (side) {
    case pTOP:
        AT.SubMatrixOf(A, 0, 0, nb, A.Cols())
        AB.SubMatrixOf(A, nb, 0, A.Rows()-nb, A.Cols())
    case pBOTTOM:
        AT.SubMatrixOf(A, 0, 0, A.Rows()-nb, A.Cols())
        AB.SubMatrixOf(A, A.Rows()-nb, 0, nb, A.Cols())
    }
}

/*
 Repartition 2 by 1 block to 3 by 1 block.
 
           AT      A0            AT       A0
 pBOTTOM: --  --> --   ; pTOP:   --  -->  A1
           AB      A1            AB       --
                   A2                     A2

 */
func repartition2x1to3x1(AT, A0, A1, A2, A *matrix.FloatMatrix, nb, pdir int) {
    nT := AT.Rows()
    if nT + nb > A.Rows() {
        nb = A.Rows() - nT
    }
    switch (pdir) {
    case pBOTTOM:
        A0.SubMatrixOf(A, 0,     0, nT, A.Cols())
        A1.SubMatrixOf(A, nT,    0, nb, A.Cols())
        A2.SubMatrixOf(A, nT+nb, 0, A.Rows()-nT-nb, A.Cols())
    case pTOP:
        A0.SubMatrixOf(A, 0,     0, nT-nb, A.Cols())
        A1.SubMatrixOf(A, nT-nb, 0, nb,    A.Cols())
        A2.SubMatrixOf(A, nT,    0, A.Rows()-nT, A.Cols())
    }
}

/*
 Continue with 2 by 1 block from 3 by 1 block.
 
           AT      A0            AT       A0
 pBOTTOM: --  <--  A1   ; pTOP:   -- <--  --
           AB      --            AB       A1
                   A2                     A2

 */
func continue3x1to2x1(AT, AB, A0, A1, A *matrix.FloatMatrix, pdir int) {
    n0 := A0.Rows()
    n1 := A1.Rows()
    switch (pdir) {
    case pBOTTOM:
        AT.SubMatrixOf(A, 0,     0, n0+n1, A.Cols())
        AB.SubMatrixOf(A, n0+n1, 0, A.Rows()-n0-n1, A.Cols())
    case pTOP:
        AT.SubMatrixOf(A, 0,  0, n0, A.Cols())
        AB.SubMatrixOf(A, n0, 0, A.Rows()-n0, A.Cols())
    }
}



/*
 Partition A to 1 by 2 blocks.

  A -->  AL | AR

 Parameter nb is initial block size for AL (pLEFT) or AR (pRIGHT).  
 */
func partition1x2(AL, AR, A *matrix.FloatMatrix, nb int, side int) {
    switch (side) {
    case pLEFT:
        AL.SubMatrixOf(A, 0, 0, A.Rows(), nb)
        AR.SubMatrixOf(A, 0, nb, A.Rows(), A.Cols()-nb)
    case pRIGHT:
        AL.SubMatrixOf(A, 0, nb, A.Rows(), A.Cols()-nb)
        AR.SubMatrixOf(A, 0, A.Cols()-1-nb, nb, A.Rows())
    }
}



/*
 Repartition 1 by 2 blocks to 1 by 3 blocks.

 pRIGHT: AL | AR  -->  A0 | A1 A2 
 pLEFT:  AL | AR  -->  A0 A1 | A2 

 Parameter As is left or right block of original 1x2 block.
 */
func repartition1x2to1x3(AL, A0, A1, A2, A *matrix.FloatMatrix, nb int, pdir int) {
    k := AL.Cols()
    if k + nb > A.Cols() {
        nb = A.Cols() - k
    }
    switch (pdir) {
    case pRIGHT:
        // A0 is AL; [A1; A2] is AR
        A0.SubMatrixOf(A, 0, 0,    A.Rows(), k)
        A1.SubMatrixOf(A, 0, k,    A.Rows(), nb)
        A2.SubMatrixOf(A, 0, k+nb, A.Rows(), A.Cols()-nb-k)
    case pLEFT:
        // A2 is AR; [A0; A1] is AL
        A0.SubMatrixOf(A, 0, 0, A.Rows(), k-nb)
        A1.SubMatrixOf(A, 0, A.Cols()-k-nb, A.Rows(), nb)
        A2.SubMatrixOf(A, 0, A.Cols()-k,    A.Rows(), A.Cols()-k)
    }
}

/*
 Repartition 1 by 2 blocks to 1 by 3 blocks.

 pRIGHT: AL | AR  --  A0 A1 | A2 
 pLEFT:  AL | AR  <--  A0 | A1 A2 

 */
func continue1x3to1x2(AL, AR, A0, A1, A *matrix.FloatMatrix, pdir int) {

    k := A0.Cols()
    nb := A1.Cols()
    switch (pdir) {
    case pRIGHT:
        // AL is [A0; A1], AR is A2
        AL.SubMatrixOf(A, 0, 0, A.Rows(), k+nb)
        AR.SubMatrixOf(A, 0, AL.Cols(), A.Rows(), A.Cols()-AL.Cols())
    case pLEFT:
        // AL is A0; AR is [A1; A2]
        if k - nb < 0 {
            nb = k
        }
        AL.SubMatrixOf(A, 0, 0, A.Rows(), k)
        AR.SubMatrixOf(A, 0, AL.Cols()-1, A.Rows(), A.Cols()-AL.Cols())
    }
}

/*
 Partition A to 2 by 2 blocks.

           ATL | ATR
  A  -->   =========
           ABL | ABR

 Parameter nb is initial block size for ATL. 
 */
func partition2x2(ATL, ATR, ABL, ABR, A *matrix.FloatMatrix, nb int, side int) {
    switch (side) {
    case pTOPLEFT:
        ATL.SubMatrixOf(A, 0, 0,  nb, nb)
        ATR.SubMatrixOf(A, 0, nb, nb, A.Cols()-nb)
        ABL.SubMatrixOf(A, nb, 0, A.Rows()-nb, nb)
        ABR.SubMatrixOf(A, nb, nb)
    case pBOTTOMRIGHT:
    }
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
    A00, A01, A02, A10, A11, A12, A20, A21, A22, A *matrix.FloatMatrix, nb int, pdir int) {

    k := ATL.Rows()
    if k + nb > A.Cols() {
        nb = A.Cols() - k
    }
    switch (pdir) {
    case pBOTTOMRIGHT:
        A00.SubMatrixOf(A, 0, 0,    k, k)
        A01.SubMatrixOf(A, 0, k,    k, nb)
        A02.SubMatrixOf(A, 0, k+nb, k, A.Cols()-k-nb)

        A10.SubMatrixOf(A, k, 0,    nb, k)
        A11.SubMatrixOf(A, k, k,    nb, nb)
        A12.SubMatrixOf(A, k, k+nb, nb, A.Cols()-k-nb)

        A20.SubMatrixOf(A, k+nb, 0,    A.Rows()-k-nb, k)
        A21.SubMatrixOf(A, k+nb, k,    A.Rows()-k-nb, nb)
        A22.SubMatrixOf(A, k+nb, k+nb)
    case pTOPLEFT:
        // move towards top left corner
    }
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
    A00, A11, A22, A *matrix.FloatMatrix, pdir int) {

    k := A00.Rows()
    mb := A11.Cols()
    switch (pdir) {
    case pBOTTOMRIGHT:
        ATL.SubMatrixOf(A, 0, 0,    k+mb, k+mb)
        ATR.SubMatrixOf(A, 0, k+mb, k+mb, A.Cols()-k-mb)

        ABL.SubMatrixOf(A, k+mb, 0, A.Rows()-k-mb, k+mb)
        ABR.SubMatrixOf(A, k+mb, k+mb)
    case pTOPLEFT:
    }
}



type pPivots struct {
    pivots []int
}

/*
 Partition p to 2 by 1 blocks.

        pT
  p --> --
        pB

 Parameter nb is initial block size for pT (pTOP) or pB (pBOTTOM).  
 */
func partitionPivot2x1(pT, pB, p *pPivots, nb, pdir int) {
    switch (pdir) {
    case pTOP:
        if nb == 0 {
            pT.pivots = nil
        } else {
            pT.pivots = p.pivots[:nb]
        }
        pB.pivots = p.pivots[nb:]
    case pBOTTOM:
        if nb > 0 {
            pT.pivots = p.pivots[:-nb]
            pT.pivots = p.pivots[len(p.pivots)-nb:]
        } else {
            pT.pivots = p.pivots
            pB.pivots = nil
        }
    }
}

/*
 Repartition 2 by 1 block to 3 by 1 block.
 
           pT      p0            pT       p0
 pBOTTOM: --  --> --   ; pTOP:   --  -->  p1
           pB      p1            pB       --
                   p2                     p2

 */
func repartPivot2x1to3x1(pT, p0, p1, p2, p *pPivots, nb, pdir int) {
    nT := len(pT.pivots)
    if nT + nb > len(p.pivots) {
        nb = len(p.pivots) - nT
    }
    switch (pdir) {
    case pBOTTOM:
        p0.pivots = pT.pivots
        p1.pivots = p.pivots[nT:nT+nb]
        p2.pivots = p.pivots[nT+nb:]
    case pTOP:
        p0.pivots = p.pivots[:nT-nb]
        p1.pivots = p.pivots[nT-nb:nT]
        p2.pivots = p.pivots[nT:]
    }
}

/*
 Continue with 2 by 1 block from 3 by 1 block.
 
           pT      p0            pT       p0
 pBOTTOM: --  <--  p1   ; pTOP:   -- <--  --
           pB      --            pB       p1
                   p2                     p2

 */
func contPivot3x1to2x1(pT, pB, p0, p1, p *pPivots, pdir int) {
    var n0, n1 int
    n0 = len(p0.pivots)
    n1 = len(p1.pivots)
    switch (pdir) {
    case pBOTTOM:
        pT.pivots = p.pivots[:n0+n1]
        pB.pivots = p.pivots[n0+n1:]
    case pTOP:
        pT.pivots = p.pivots[:n0]
        pB.pivots = p.pivots[n0:]
    }
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
