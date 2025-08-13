#noncvxui(x1, x2, x3) = (x1 + x2 + x3)^2 + 4cos(x1 + x2 + x3)

function noncvxu2(x, cache)
	check_x_indices(x, cache)
	n = length(x)
	return sumi(i -> @inbounds(noncvxui(x[i], x[(3i - 2) % n + 1], x[(7i - 3) % n + 1])), 
		cache, eachindex(x))
end

function noncvxu2_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	n = length(x)
	gradient .= 0
	@inbounds for i in eachindex(x)
		i2 = (3i - 2) % n + 1
		i3 = (7i - 3) % n + 1
		xi = x[i]
		xi2 = x[i2]
		xi3 = x[i3]
		a = 2(xi + xi2 + xi3) - 4sin(xi + xi2 + xi3)
		gradient[i] += a
		gradient[i2] += a
		gradient[i3] += a
	end
	return gradient
end

"""
NAME          NONCVXU2

*   Problem :
*   *********

*   A nonconvex unconstrained function with a unique minimum value

*   SIF input: Nick Gould, April 1996

*   classification OUR2-AN-V-0

*   The number of variables

*IE N                   10             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
*IE N                   100000         -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_noncvxu2(; n = 5000, T = typeof(1.))
    return (x0 = collect(T, 1:n), obj = x -> noncvxu2(x, Vector{T}(undef, n)), grad! = noncvxu2_grad!)
end