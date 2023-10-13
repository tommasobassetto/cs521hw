; Lines that start with a semicolon are comments

; zonotope Z constants
(declare-const a1 Real)
(declare-const b1 Real)
(declare-const c1 Real)
(declare-const d1 Real)

(declare-const a2 Real)
(declare-const b2 Real)
(declare-const c2 Real)
(declare-const d2 Real)

; vertices of Z1 - 6,2
(declare-const z1v1x Int)
(assert (= z1v1x 6))

(declare-const z1v1y Real)
(assert (= z1v1y 2))

; noise symbols
(declare-const e1z1v1 Real)
(assert (<= e1z1v1 1))
(assert (>= e1z1v1 -1))

(declare-const e2z1v1 Real)
(assert (<= e2z1v1 1))
(assert (>= e2z1v1 -1))

(declare-const e3z1v1 Real)
(assert (<= e3z1v1 1))
(assert (>= e3z1v1 -1))

; require it to be in the zonotope
(assert (= z1v1x (+ a1 (* b1 e1z1v1) (* c1 e2z1v1) (* d1 e3z1v1))))
(assert (= z1v1y (+ a2 (* b2 e1z1v1) (* c2 e2z1v1) (* d2 e3z1v1))))

; vertices of Z1 - 0,-2
(declare-const z1v2x Real)
(assert (= z1v2x 0))

(declare-const z1v2y Real)
(assert (= z1v2y -2))

; noise symbols
(declare-const e1z1v2 Real)
(assert (<= e1z1v2 1))
(assert (>= e1z1v2 -1))

(declare-const e2z1v2 Real)
(assert (<= e2z1v2 1))
(assert (>= e2z1v2 -1))

(declare-const e3z1v2 Real)
(assert (<= e3z1v2 1))
(assert (>= e3z1v2 -1))

; require it to be in the zonotope
(assert (= z1v2x (+ a1 (* b1 e1z1v2) (* c1 e2z1v2) (* d1 e3z1v2))))
(assert (= z1v2y (+ a2 (* b2 e1z1v2) (* c2 e2z1v2) (* d2 e3z1v2))))

; vertices of Z1 - 2, 0
(declare-const z1v3x Real)
(assert (= z1v3x 2))

(declare-const z1v3y Real)
(assert (= z1v3y 0))

; noise symbols
(declare-const e1z1v3 Real)
(assert (<= e1z1v3 1))
(assert (>= e1z1v3 -1))

(declare-const e2z1v3 Real)
(assert (<= e2z1v3 1))
(assert (>= e2z1v3 -1))

(declare-const e3z1v3 Real)
(assert (<= e3z1v3 1))
(assert (>= e3z1v3 -1))

; require it to be in the zonotope
(assert (= z1v3x (+ a1 (* b1 e1z1v3) (* c1 e2z1v3) (* d1 e3z1v3))))
(assert (= z1v3y (+ a2 (* b2 e1z1v3) (* c2 e2z1v3) (* d2 e3z1v3))))

; vertices of Z1 - 4, 0
(declare-const z1v4x Real)
(assert (= z1v4x 4))

(declare-const z1v4y Real)
(assert (= z1v4y 0))

; noise symbols
(declare-const e1z1v4 Real)
(assert (<= e1z1v4 1))
(assert (>= e1z1v4 -1))

(declare-const e2z1v4 Real)
(assert (<= e2z1v4 1))
(assert (>= e2z1v4 -1))

(declare-const e3z1v4 Real)
(assert (<= e3z1v4 1))
(assert (>= e3z1v4 -1))

; require it to be in the zonotope
(assert (= z1v4x (+ a1 (* b1 e1z1v4) (* c1 e2z1v4) (* d1 e3z1v4))))
(assert (= z1v4y (+ a2 (* b2 e1z1v4) (* c2 e2z1v4) (* d2 e3z1v4))))

; vertices of Z2 - 0, 2
(declare-const z2v1x Real)
(assert (= z2v1x 0))

(declare-const z2v1y Real)
(assert (= z2v1y 2))

; noise symbols
(declare-const e1z2v1 Real)
(assert (<= e1z2v1 1))
(assert (>= e1z2v1 -1))

(declare-const e2z2v1 Real)
(assert (<= e2z2v1 1))
(assert (>= e2z2v1 -1))

(declare-const e3z2v1 Real)
(assert (<= e3z2v1 1))
(assert (>= e3z2v1 -1))

; require it to be in the zonotope
(assert (= z2v1x (+ a1 (* b1 e1z2v1) (* c1 e2z2v1) (* d1 e3z2v1))))
(assert (= z2v1y (+ a2 (* b2 e1z2v1) (* c2 e2z2v1) (* d2 e3z2v1))))

; vertices of Z2 - -2, 0
(declare-const z2v2x Real)
(assert (= z2v2x -2))

(declare-const z2v2y Real)
(assert (= z2v2y 0))

; noise symbols
(declare-const e1z2v2 Real)
(assert (<= e1z2v2 1))
(assert (>= e1z2v2 -1))

(declare-const e2z2v2 Real)
(assert (<= e2z2v2 1))
(assert (>= e2z2v2 -1))

(declare-const e3z2v2 Real)
(assert (<= e3z2v2 1))
(assert (>= e3z2v2 -1))

; require it to be in the zonotope
(assert (= z2v2x (+ a1 (* b1 e1z2v2) (* c1 e2z2v2) (* d1 e3z2v2))))
(assert (= z2v2y (+ a2 (* b2 e1z2v2) (* c2 e2z2v2) (* d2 e3z2v2))))

; vertices of Z2 - 4, 0
(declare-const z2v3x Real)
(assert (= z2v3x 0))

(declare-const z2v3y Real)
(assert (= z2v3y 2))

; noise symbols
(declare-const e1z2v3 Real)
(assert (<= e1z2v3 1))
(assert (>= e1z2v3 -1))

(declare-const e2z2v3 Real)
(assert (<= e2z2v3 1))
(assert (>= e2z2v3 -1))

(declare-const e3z2v3 Real)
(assert (<= e3z2v3 1))
(assert (>= e3z2v3 -1))

; require it to be in the zonotope
(assert (= z2v3x (+ a1 (* b1 e1z2v3) (* c1 e2z2v3) (* d1 e3z2v3))))
(assert (= z2v3y (+ a2 (* b2 e1z2v3) (* c2 e2z2v3) (* d2 e3z2v3))))

; vertices of Z2 - 2, -2
(declare-const z2v4x Real)
(assert (= z2v4x 0))

(declare-const z2v4y Real)
(assert (= z2v4y 2))

; noise symbols
(declare-const e1z2v4 Real)
(assert (<= e1z2v4 1))
(assert (>= e1z2v4 -1))

(declare-const e2z2v4 Real)
(assert (<= e2z2v4 1))
(assert (>= e2z2v4 -1))

(declare-const e3z2v4 Real)
(assert (<= e3z2v4 1))
(assert (>= e3z2v4 -1))

; require it to be in the zonotope
(assert (= z2v4x (+ a1 (* b1 e1z2v4) (* c1 e2z2v4) (* d1 e3z2v4))))
(assert (= z2v4y (+ a2 (* b2 e1z2v4) (* c2 e2z2v4) (* d2 e3z2v4))))

; This command asks the solver to check the satisfiability of your query
; If you wrote a sound transformer, the solver should say 'unsat'
(check-sat)
; If the solver returns 'sat', uncommenting the line below will give you the values of the various variables that violate the soundness property. This will help you debug your solution.
(get-model)
