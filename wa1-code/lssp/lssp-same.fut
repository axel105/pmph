-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input { [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1] }
-- output { 5 }
-- compiled input { [1, 8, 8, 9, 8, 8, 10, 8, 8] }
-- output { 2 }
-- compiled input { [1, 2, 3, 4, 3, 2, 1] }
-- output { 1 }
-- compiled input { [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] }
-- output { 10 }
-- compiled input { [1, 1] }
-- output { 2 }
-- compiled input { [1, 1, 1] }
-- output { 3 }
-- compiled input { [1, 1, 1, 1] }
-- output { 4 }
-- compiled input { [1, 1, 1, 1, 1] }
-- output { 5 }


import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x == y)
  in  lssp_seq pred1 pred2 xs
--  in  lssp pred1 pred2 xs
