genrose(x) = sumi(i -> 100(x[i] - x[i-1]^2)^2 + (x[i] - 1)^2, zero(eltype(x)), dindices(x, 1,0))

function genrose_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	a = al = zero(eltype(x))
    for i in dindices(x, 1,0)
		al = a
		a = 202x[i] - 200x[i-1]^2 - 2
        gradient[i-1] = -400(x[i] - x[i-1]^2) * x[i - 1] + al
    end
	gradient[end] = a
    return gradient
end

"""
NAME          GENROSE

*   Problem :
*   --------

*   The generalized Rosenbrock function.

*   Source: problem 5 in
*   S. Nash,
*   "Newton-type minimization via the Lanczos process",
*   SIAM J. Num. Anal. 21, 1984, 770-788.

*   SIF input: Nick Gould, Oct 1992.
*              minor correction by Ph. Shott, Jan 1995.

*   classification SUR2-AN-V-0

*   Number of variables

*IE N                   5              -PARAMETER
*IE N                   10             -PARAMETER
*IE N                   100            -PARAMETER
 IE N                   500            -PARAMETER
*IE N                   10             -PARAMETER
...

Keyword argument:
; n = 500, T = typeof(1.)
"""
function gen_genrose(; n = 500, T = typeof(1.))
	return (x0 = T[i/(n+1) for i in 1:n], 
			obj = genrose, 
			grad! = genrose_grad!)
end