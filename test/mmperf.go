
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package main

import (
    "github.com/hrautila/matrix"
    "github.com/hrautila/linalg/blas"
    "github.com/hrautila/linalg"
    "github.com/hrautila/mperf"
    "github.com/hrautila/matops/calgo"
    "fmt"
    "flag"
    "runtime"
    "strings"
    "strconv"
    "unsafe"
)


var M, N, P, MB, NB int
var randomData bool
var check bool
var verbose bool
var asGflops bool
var singleTest bool
var nWorker int
var testName string
var testCount int
var VPsize int
var sizeList string

func init() {
    flag.IntVar(&M, "M", 600, "Matrix A rows.")
    flag.IntVar(&N, "N", 600, "Matrix B cols.")
    flag.IntVar(&P, "P", 600, "Matrix A cols, B rows.")
    flag.IntVar(&MB, "MB", 64, "Row blocking size.")
    flag.IntVar(&NB, "NB", 64, "Column blocking size.")
    flag.IntVar(&nWorker, "W", 2, "Number of workers for parallel runs")
    flag.IntVar(&VPsize, "H", 64, "Viewport size.")
    flag.BoolVar(&check, "C", false, "Check result against reference (gemm).")
    flag.BoolVar(&verbose, "v", false, "Be verbose.")
    flag.BoolVar(&asGflops, "g", false, "Report as Gflops.")
    flag.BoolVar(&randomData, "R", true, "Generate random data.")
    flag.BoolVar(&singleTest, "s", false, "Run single test run for given matrix sizes.")
    flag.IntVar(&testCount, "n", 5, "Number of test runs.")
    flag.StringVar(&testName, "T", "test", "Test name for reporting")
    flag.StringVar(&sizeList, "L", "", "Comma separated list of sizes.")
}

var sizes []int = []int{
    10, 30, 50, 70, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500}

func index(i, r, sz int) int {
    return i*sz/r - ((i*sz/r) & 0x1);
}

func TestTemplate(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A = matrix.FloatNormal(m, p)
    B = matrix.FloatNormal(p, n)
    C = matrix.FloatZeros(m, n)
    fnc = func() {
        // test core here
    }
    return
}

func CTestMultUnAligned(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)
    fnc = func() {
        Ar := A.FloatArray()
        Br := B.FloatArray()
        Cr := C.FloatArray()
        ldC := C.LeadingIndex()
        ldA := A.LeadingIndex()
        ldB := B.LeadingIndex()
        calgo.MultUnAligned(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB, p, 0, n, 0, m, VPsize, NB, MB)
    }
    return fnc, A, B, C
}

func CTestMultAligned(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)
    fnc = func() {
        Ar := A.FloatArray()
        Br := B.FloatArray()
        Cr := C.FloatArray()
        ldC := C.LeadingIndex()
        ldA := A.LeadingIndex()
        ldB := B.LeadingIndex()
        C_aligned := uintptr(unsafe.Pointer(&Cr[0])) % 16 == 0
        A_aligned := uintptr(unsafe.Pointer(&Ar[0])) % 16 == 0
        B_aligned := uintptr(unsafe.Pointer(&Br[0])) % 16 == 0
        if ! (C_aligned && A_aligned && B_aligned) {
            fmt.Printf("C aligned: %v\nA aligned: %v\nB aligned: %v\n", C_aligned,
                A_aligned, B_aligned)
            return
        }
        calgo.MultAligned(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB, p, 0, n, 0, m, VPsize, NB, MB)
    }
    return fnc, A, B, C
}

func CTestMultUnAlignedTransA(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)
    fnc = func() {
        Ar := A.FloatArray()
        Br := B.FloatArray()
        Cr := C.FloatArray()
        ldC := C.LeadingIndex()
        ldA := A.LeadingIndex()
        ldB := B.LeadingIndex()
        calgo.MultUnAlignedTransA(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB, p, 0, n, 0, m, VPsize, NB, MB)
    }
    return fnc, A, B, C
}

func CTestMultAlignedTransA(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)
    fnc = func() {
        Ar := A.FloatArray()
        Br := B.FloatArray()
        Cr := C.FloatArray()
        ldC := C.LeadingIndex()
        ldA := A.LeadingIndex()
        ldB := B.LeadingIndex()
        C_aligned := uintptr(unsafe.Pointer(&Cr[0])) % 16 == 0
        A_aligned := uintptr(unsafe.Pointer(&Ar[0])) % 16 == 0
        B_aligned := uintptr(unsafe.Pointer(&Br[0])) % 16 == 0
        if ! (C_aligned && A_aligned && B_aligned) {
            fmt.Printf("C aligned: %v\nA aligned: %v\nB aligned: %v\n", C_aligned,
                A_aligned, B_aligned)
            return
        }
        calgo.MultAlignedTransA(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB, p, 0, n, 0, m, VPsize, NB, MB)
    }
    return fnc, A, B, C
}



func PTestAligned(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)

    fnc = func() {
        Ar := A.FloatArray()
        Br := B.FloatArray()
        Cr := C.FloatArray()
        ldA := A.LeadingIndex()
        ldB := B.LeadingIndex()
        ldC := C.LeadingIndex()

        worker := func(m0, n0, p0, start, end int, ready chan int) {
            calgo.MultAligned(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB,
                p0, 0, n, start, end, VPsize, NB, MB)
            ready <- 1
        }
        // fire up nWorker-1 go-routines
        ch := make(chan int, nWorker-1)
        for k := 1; k < nWorker; k++ {
            go worker(m, n, p, index(k, nWorker, m), index((k+1), nWorker, m), ch)
        }
        calgo.MultAligned(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB,
            p, 0, n, 0, index(1, nWorker, m), VPsize, NB, MB)
        nready := 1
        for nready < nWorker {
            nready += <- ch
        }
    }
    return fnc, A, B, C
}

func PTestUnAligned(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)

    fnc = func() {
        Ar := A.FloatArray()
        Br := B.FloatArray()
        Cr := C.FloatArray()
        ldA := A.LeadingIndex()
        ldB := B.LeadingIndex()
        ldC := C.LeadingIndex()

        worker := func(m0, n0, p0, start, end int, ready chan int) {
            calgo.MultUnAligned(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB,
                p0, 0, n, start, end, VPsize, NB, MB)
            ready <- 1
        }
        // fire up nWorker-1 go-routines
        ch := make(chan int, nWorker-1)
        for k := 1; k < nWorker; k++ {
            go worker(m, n, p, index(k, nWorker, m), index((k+1), nWorker, m), ch)
        }
        calgo.MultUnAligned(Cr, Ar, Br, 1.0, 1.0, ldC, ldA, ldB,
            p, 0, n, 0, index(1, nWorker, m), VPsize, NB, MB)
        nready := 1
        for nready < nWorker {
            nready += <- ch
        }
    }
    return fnc, A, B, C
}

func CTestGemm(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)
    fnc = func() {
        blas.GemmFloat(A, B, C, 1.0, 1.0)
    }
    return fnc, A, B, C
}

func CTestGemmTransA(m, n, p int) (fnc func(), A, B, C *matrix.FloatMatrix) {
    A, B, C = mperf.MakeData(m, n, p, randomData, false)
    fnc = func() {
        blas.GemmFloat(A, B, C, 1.0, 1.0, linalg.OptTransA)
    }
    return fnc, A, B, C
}

var tests map[string]mperf.MatrixTestFunc = map[string]mperf.MatrixTestFunc{
    "ParallelAligned": PTestAligned,
    "ParallelUnAligned": PTestUnAligned,
    "MultUnAligned": CTestMultUnAligned,
    "MultAligned": CTestMultAligned,
    "MultUnAlignedTransA": CTestMultUnAligned,
    "MultAlignedTransA": CTestMultAligned,
    "GemmTransA": CTestGemmTransA,
    "Gemm": CTestGemm}

    
func parseSizeList(s string) []int {
    sl := strings.Split(s, ",")
    il := make([]int, 0)
    for _, snum := range sl {
        n, err := strconv.ParseInt(snum, 0, 32)
        if err == nil {
            il = append(il, int(n))
        }
    }
    return il
}

func main() {
    flag.Parse()
    runtime.GOMAXPROCS(nWorker)
    testFunc, ok := tests[testName]
    if ! ok {
        fmt.Printf("Error: test %s does not exists.\nKnown tests:\n", testName)
        for tname := range tests {
            fmt.Printf("\t%s\n", tname)
        }
        return
    }
    if singleTest {
        sec, _ := mperf.SingleTest(testName, testFunc, M, N, P, check, verbose)
        fmt.Printf("%vs\n", sec)
    } else {
        if len(sizeList) > 0 {
            sizes = parseSizeList(sizeList)
        }
        times := mperf.MultipleSizeTests(testFunc, sizes, testCount, verbose)
        if asGflops {
            if verbose {
                fmt.Printf("calculating Gflops ...\n")
            }
            for sz := range times {
                n := int64(sz)
                times[sz] = 2.0*float64(n*n*n) / times[sz] * 1e-9
            }
        }
        // print out as python dictionary
        fmt.Printf("{")
        i := 0
        for sz := range times {
            if i > 0 {
                fmt.Printf(", ")
            }
            fmt.Printf("%d: %v", sz, times[sz])
            i++
        }
        fmt.Printf("}\n")
    }
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:
