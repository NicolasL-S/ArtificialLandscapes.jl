function luksan17ls(x, sinx, cosx, y)
    check_x_indices(x, sinx)
    @. sinx = sin(x)
    @. cosx = cos(x)
    s = zero(eltype(x))
	for i in 0:2:96, l in 1:4
		s += sumi(q -> @inbounds(l * q^2 * sinx[i + q] - l^2 * q * cosx[i + q]), y[l], 1:4)^2
    end
    return s
end

function luksan17ls_grad!(gradient, x, sinx, cosx, y)
    check_x_indices(x, sinx)
    check_gradient_indices(gradient, x)
    @. sinx = sin(x)
    @. cosx = cos(x)
	gradient .= 0
	for i in 0:2:96, l in 1:4
		a = 2sumi(q -> @inbounds(l * q^2 * sinx[i + q] - l^2 * q * cosx[i + q]), y[l], 1:4)
		@inbounds for q in 1:4
			gradient[i + q] += a * (l * q^2 * cosx[i + q] + l^2 * q * sinx[i + q])
		end
    end
    return gradient
end

"""
NAME          LUKSAN17LS

*   Problem :
*   *********

*   Problem 17 (sparse trigonometric) in the paper

*     L. Luksan
*     Hybrid methods in large sparse nonlinear least squares
*     J. Optimization Theory & Applications 89(3) 575-595 (1996)

*   SIF input: Nick Gould, June 2017.

*   least-squares version

*   classification SUR2-AN-V-0
...

Keyword argument:
; n = 100, T = typeof(1.)
"""
function gen_luksan17ls(; n = 100, T = typeof(1.))
    y = T[30.6, 72.2, 124.4, 187.4]
	sinx = Vector{T}(undef, n)
	cosx = Vector{T}(undef, n)
	return (x0 = repeat(T[-0.8,1.2,-1.2,0.8]; outer = n ÷ 4), 
			obj = x -> luksan17ls(x, sinx, cosx, y), 
			grad! = (gradient, x) -> luksan17ls_grad!(gradient, x, sinx, cosx, y))
end