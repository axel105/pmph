-- Parallel Longest Satisfying Segment
--
-- ==
--compiled input { [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1] }
--output { 5 }
--compiled input { [1i32, -2, -2, 0, 3, 0, 3, 0, 3, 4, -6, 1] }
--output { 1 }
--compiled input { [0, 0, 0, 0, 0, 0] }
--output { 6 }
--compiled input { [0] }
--output { 1 }
--compiled input { [1, 2, 3, 4, 5] }
--output { 0 }
--compiled input { [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
--output { 10 }


import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
  in  lssp_seq pred1 pred2 xs
--  in  lssp pred1 pred2 xs
