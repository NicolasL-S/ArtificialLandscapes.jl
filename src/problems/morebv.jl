function morebv(x, cache)
    check_x_indices(x, cache)
	h = 1 / eltype(x)(length(x) + 1)
	half_h_sq = h^2 / 2
	s = (2x[begin] - x[begin + 1] + half_h_sq * (x[begin] + 1)^3)^2
	s += sumi(i -> @inbounds(2x[i] - x[i - 1] - x[i + 1] + half_h_sq * (x[i] + i * h + 1)^3)^2, 
		cache, dindices(x,1,-1)) 
	s += (2x[end] - x[end - 1] + half_h_sq * (x[end] + length(x) * h + 1)^3)^2
	return s
end

function morebv_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	n = length(x)
	h = 1 / eltype(x)(n + 1)
	half_h_sq = h^2 / 2
	gradient .= 0
	a = 2(2x[begin] - x[begin + 1] + half_h_sq * (x[begin] + 1)^3)
	gradient[begin] = a * (2 + half_h_sq * 3(x[begin] + 1)^2)
	gradient[begin + 1] = -a
	for i in dindices(x,1,-1)
		a = 2(2x[i] - x[i - 1] - x[i + 1] + half_h_sq * (x[i] + i * h + 1)^3)
		gradient[i - 1] -= a
		gradient[i] += a * (2 + half_h_sq * 3(x[i] + i * h + 1)^2)
		gradient[i + 1] = -a
	end
	a = 2(2x[end] - x[end - 1] + half_h_sq * (x[end] + n * h + 1)^3)
	gradient[end - 1] -= a
	gradient[end] += a * (2 + half_h_sq * 3(x[end] + n * h + 1)^2)
	return gradient
end

"""
NAME          MOREBV

*   Problem :
*   *********

*   The Boundary Value problem.
*   This is the nonlinear least-squares version without fixed variables.

*   Source:  problem 28 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#17 (p. 75).

*   SIF input: Ph. Toint, Dec 1989 and Nick Gould, Oct 1992.
*              correction by S. Gratton & Ph. Toint, May 2024

*   classification SUR2-MN-V-0

*   The number of variables is N.

*IE N                   10             -PARAMETER     original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
 ...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_morebv(; n = 5000, T = typeof(1.))
    return (x0 = [i/(1 + n)*(i/(1 + n) - 1) for i in 1:n], 
            obj = x -> morebv(x, Vector{T}(undef, n)), 
            grad! = (gradient, x) -> morebv_grad!(gradient, x))
end