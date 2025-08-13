function fletchcr(x)
	length(x) % 2 == 0 || throw(ArgumentError("Length of x needs to be even."))
	return sumi(i -> @inbounds(100(x[i+1] - x[i]^2)^2 + (1 - x[i])^2), zero(eltype(x)), dindices(x, 0,-1))
end

function fletchcr_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	a = zero(eltype(x))
	@inbounds for i in dindices(x, 0,-1)
		a_now = 200(x[i + 1] - x[i]^2)
		gradient[i] = -2((a_now - 1) * x[i] + 1) + a
		a = a_now
	end
	gradient[end] = a
	return gradient
end

"""
NAME          FLETCHCR

*   Problem :
*   --------

*   The chained Rosenbrock function as given by Fletcher.

*   Source:  The second problem given by
*   R. Fletcher,
*   "An optimal positive definite update for sparse Hessian matrices"
*   Numerical Analysis report NA/145, University of Dundee, 1992.

*   SIF input: Nick Gould, Oct 1992.

*   classification OUR2-AN-V-0

*   The Number of variables is N.

*IE N                   10             -PARAMETER     original value
*IE N                   100            -PARAMETER
 IE N                   1000           -PARAMETER
...

Keyword arguments:
; n = 1000, T = typeof(1.)
"""
gen_fletchcr(; n = 1000, T = typeof(1.)) = (x0 = zeros(T, n), obj = fletchcr, grad! = fletchcr_grad!)