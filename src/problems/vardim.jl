function vardim(x)
	n = length(x)
	T0 = zero(eltype(x))
	s = sum(xi -> (xi - 1)^2, x)
	s2 = (sumi(i -> i*x[i], T0, eachindex(x)) - n * (n + 1)/2)^2
	return s + s2 + s2^2
end

function vardim_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	n = length(x)
	T0 = zero(eltype(x))
	s2 = sumi(i -> i*x[i], T0, eachindex(x)) - n * (n + 1)/2
	a = 2s2 + 4s2^3
	@inbounds for i in eachindex(x)
		gradient[i] = 2(x[i] - 1) + a * i
	end
	return gradient
end

"""
NAME          VARDIM

*   Problem :
*   *********

*   Variable dimension problem
*   This problem is a sum of n+2 least-squares groups, the first n of
*   which have only a linear element.
*   It Hessian matrix is dense.

*   Source:  problem 25 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#72 (p.98).

*   SIF input: Ph. Toint, Dec 1989.

*   classification  OUR2-AN-V-0

*   N is the number of free variables

*IE N                   10             -PARAMETER     original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
 IE N                   200            -PARAMETER
 ...

Keyword argument:
; n = 200, T = typeof(1.)
"""
function gen_vardim(; n = 200, T = typeof(1.))
	return (x0 = 1 .- T.(collect(1:n)) ./ n, obj = vardim, grad! = vardim_grad!)
end