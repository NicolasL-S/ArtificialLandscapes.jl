function oscipath(x)
	length(x) < 2 && throw(BoundsError("x must be at least length 2"))
	return sumi(i -> @inbounds(500(x[i + 1] - 2x[i]^2 + 1)^2), (x[begin] - 1)^2 / 4, dindices(x, 0, -1))
end

function oscipath_grad!(gradient, x)
	length(x) < 2 && throw(BoundsError("x must be at least length 2"))
    check_gradient_indices(gradient, x)
	a = (x[begin] - 1) / 2
	a_old = zero(eltype(x))
	@inbounds for i in dindices(x, 0, -1)
		a_old = a
		a = 1000(x[i + 1] - 2x[i]^2 + 1)
		gradient[i] = a * (-4x[i]) + a_old
	end
	gradient[end] = a
	return gradient
end

"""
NAME          OSCIPATH

*   Problem :
*   *********

*   An "oscillating path" problem due to Yurii Nesterov

*   SIF input: Nick Gould, Dec 2006.

*   classification SUR2-AN-V-0

*   Number of variables

*IE N                   2              -PARAMETER
*IE N                   5              -PARAMETER
*IE N                   10             -PARAMETER
*IE N                   25             -PARAMETER
*IE N                   100            -PARAMETER
 IE N                   500            -PARAMETER

*   the weight factor

*RE RHO                 1.0            -PARAMETER    Nesterov's original value
 RE RHO                 500.0          -PARAMETER    Florian Jarre's value
 ...

Keyword argument:
; n = 500, T = typeof(1.)
"""
function gen_oscipath(;n = 500, T = typeof(1.))
	x0  = ones(T, n)
	x0[begin] = -1
    return (x0 = x0, obj = oscipath, grad! = oscipath_grad!)
end