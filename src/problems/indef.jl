function indef(x, cache)
    check_x_indices(x, cache)
    return sum(x) + sumi(i -> cos(2x[i] - x[end] - x[begin]) / 2, cache, dindices(x, 1,-1))
end

function indef_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	gbegin = gend = zero(eltype(x))
	for i in dindices(x, 1,-1)
		a = -0.5sin(2x[i] - x[end] - x[begin])
		gbegin -= a
		gend -= a
		gradient[i] = 2a + 1
	end
	gradient[end] = gend + 1
	gradient[begin] = gbegin + 1
	return gradient
end

"""
NAME          INDEF

*   Problem :
*   *********

*   A nonconvex problem which has an indefinite Hessian at
*   the starting point.

*   SIF input: Nick Gould, Oct 1992.

*   classification OUR2-AN-V-0

*   The number of variables is N.

*IE N                   10             -PARAMETER     
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   5000           -PARAMETER
 ...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_indef(; n = 5000, T = typeof(1.))
	return (x0 = T[i/(n + 1) for i in 1:n], 
			obj = x -> indef(x, Vector{T}(undef, n)), 
			grad! = indef_grad!)
end