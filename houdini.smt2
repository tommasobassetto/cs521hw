; define the loop
(define-fun loopfn_i ((i Int) (tmp Int) (x Int) (y Int) (z Int)) Int
    (ite (< i 10) (+ i 1) i)
)

(define-fun loopfn_x ((i Int) (tmp Int) (x Int) (y Int) (z Int)) Int
    (ite (< i 10) y x)
)

(define-fun loopfn_y ((i Int) (tmp Int) (x Int) (y Int) (z Int)) Int
    (ite (< i 10) z y)
)

; We can use x instead of tmp as this is the x from the start of the iteration
(define-fun loopfn_z ((i Int) (tmp Int) (x Int) (y Int) (z Int)) Int
    (ite (< i 10) x z)
)

(define-fun loopfn_tmp ((i Int) (tmp Int) (x Int) (y Int) (z Int)) Int
    (ite (< i 10) x tmp)
)

; assert recursive condition requirements
; i = 0
;(assert (forall 
;    ((in Int) (xn Int) (yn Int) (zn Int) (tmpn Int))
;    (=> (= in 0) (= (loopfn_i in tmpn xn yn zn) 0))
;))

; i <= 10
;(assert (forall 
;    ((in Int) (xn Int) (yn Int) (zn Int) (tmpn Int))
;    (=> (<= in 10) (<= (loopfn_i in tmpn xn yn zn) 10))
;))

; i > 10
; this will retuern SAT as this condition is true recursively
; but this does not match with our initial condition
;(assert (forall 
;    ((in Int) (xn Int) (yn Int) (zn Int) (tmpn Int))
;    (=> (> in 10) (> (loopfn_i in tmpn xn yn zn) 10))
;))

; x != y
;(assert (forall 
;    ((in Int) (xn Int) (yn Int) (zn Int) (tmpn Int))
;    (=> (distinct xn yn) (distinct (loopfn_x in tmpn xn yn zn) (loopfn_y in tmpn xn yn zn)))
;))

; x != y, y != z, x != z
(assert (forall 
    ((in Int) (xn Int) (yn Int) (zn Int) (tmpn Int))
    (=> (distinct xn yn zn tmpn) 
        (distinct 
            (loopfn_x in tmpn xn yn zn)
            (loopfn_y in tmpn xn yn zn)
            (loopfn_z in tmpn xn yn zn)
        )
    )
))


; This command asks the solver to check the satisfiability of your query
; If you wrote a sound transformer, the solver should say 'unsat'
(check-sat)
; If the solver returns 'sat', uncommenting the line below will give you the values of the various variables that violate the soundness property. This will help you debug your solution.
;(get-model)
