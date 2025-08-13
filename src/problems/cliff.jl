cliff(x) = (eltype(x)(0.01) * x[1] - eltype(x)(0.03))^2 - x[1] + x[2] + exp(20(x[1] - x[2]))

function cliff_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	T = eltype(x)
    a = exp(20(x[1] - x[2]))
    gradient[1] = T(0.0002) * x[1] - T(1.0006) + 20a
    gradient[2] = 1 - 20a
    return gradient
end

"""
NAME          CLIFF

*   Problem :
*   *********

*   The "cliff problem" in 2 variables

*   Source:  problem 206 (p. 46) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AN-2-0
...

Keyword arguments
; T = typeof(1.)
"""
gen_cliff(; T = typeof(1.)) = (x0 = T[0, -1], obj = cliff, grad! = cliff_grad!)