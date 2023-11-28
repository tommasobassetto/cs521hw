; Lines that start with a semicolon are comments

; Define the function for which you are constructing a transformer

(define-fun abs ((x Real)) Real
(ite (> x 0) x (- x))       ; absolute value function
)


(define-fun max ((l Real) (u Real)) Real
(ite (> l u) l u)       ; max value function
)

(define-fun min ((l Real) (u Real)) Real
(ite (> l u) u l)       ; min value function
)

(define-fun pow2 ((x Real)) Real
(* x x) ; x^2
)
(define-fun pow3 ((x Real)) Real
(* x (pow2 x)) ; x^3
)

(define-fun sgneq ((l Real) (u Real)) Bool
(ite (> (* l u) 0) true false)       ; “sign equals” function
)


(define-fun f ((x Real)) Real
(max (pow2 x) (pow3 x)) ; max(x^2, x^3)
)


(define-fun Tf_lower ((l Real) (u Real)) Real
(ite (sgneq l u) (min (f l) (f u)) 0) ; lower bound of f(x) on [l, u]
)

(define-fun Tf_upper ((l Real) (u Real)) Real
(max (f l) (f u)) ; upper bound of f(x) on [l, u]
)


; To state the correctness of the transformer, ask the solver if there is 
; (1) a Real number x and (2) an interval [l,u]
; that violate the soundness property, i.e., satisfy the negation of the soundness property.

(declare-const x Real)
(declare-const l Real)
(declare-const u Real)

; store complex expressions in intermediate variables
; output under the function
(declare-const fx Real)
(assert (= fx (f x)))
; lower bound of range interval
(declare-const l_Tf Real)
(assert (= l_Tf (Tf_lower l u)))
; upper bound of range interval
(declare-const u_Tf Real)
(assert (= u_Tf (Tf_upper l u)))


(assert (not                         ; negation of soundness property 
(=>  
    (and (<= l x) (<= x u))          ; if input is within given bounds
    (and (<= l_Tf fx) (<= fx u_Tf))  ; then output is within transformer bounds
)))


; This command asks the solver to check the satisfiability of your query
; If you wrote a sound transformer, the solver should say 'unsat'
(check-sat)
; If the solver returns 'sat', uncommenting the line below will give you the values of the various variables that violate the soundness property. This will help you debug your solution.
;(get-model)
