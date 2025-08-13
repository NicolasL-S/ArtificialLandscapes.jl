function luksan13ls(x)
    check_x_indices(x, 98; regenerate = false)
    s = zero(eltype(x))
	@inbounds @simd for k in 0:31
        i = 3k + 1
		s += 100(x[i]^2 - x[i + 1])^2
		s += 100(x[i + 1]^2 - x[i + 2])^2
		s += ((x[i + 2] - x[i + 3])^2)^2
		s += ((x[i + 3] - x[i + 4])^2)^2
		s += (x[i] + x[i + 1]^2 + x[i + 2] - 30)^2
		s += (x[i + 1] - x[i + 2]^2 + x[i + 3] - 10)^2
		s += (x[i] * x[i + 4] - 10)^2
    end
    return s
end

function luksan13ls_grad!(gradient, x)
    check_x_indices(x, 98; regenerate = false)
    check_gradient_indices(gradient, x)
	gradient .= 0
	@inbounds for k in 0:31
        i = 3k + 1
		a = 200(x[i]^2 - x[i + 1])
		b = 200(x[i + 1]^2 - x[i + 2])
		c = 4(x[i + 2] - x[i + 3])^3
		d = 4(x[i + 3] - x[i + 4])^3
		e = 2(x[i] + x[i + 1]^2 + x[i + 2] - 30)
		f = 2(x[i + 1] - x[i + 2]^2 + x[i + 3] - 10)
		g = 2(x[i] * x[i + 4] - 10)
		gradient[i] += a * 2x[i] + e + g * x[i + 4]
		gradient[i + 1] += -a + b * 2x[i+1] + e * 2x[i + 1] + f
		gradient[i + 2] += -b + c + e - f * 2x[i + 2]
		gradient[i + 3] += -c + d + f
		gradient[i + 4] += -d + g * x[i]
    end
    return gradient
end

"""
NAME          LUKSAN13LS

*   Problem :
*   *********

*   Problem 13 (chained and modified HS48) in the paper

*     L. Luksan
*     Hybrid methods in large sparse nonlinear least squares
*     J. Optimization Theory & Applications 89(3) 575-595 (1996)

*   SIF input: Nick Gould, June 2017.

*   least-squares version

*   classification SUR2-AN-V-0
...

Keyword argument:
; n = 98, T = typeof(1.)
"""
function gen_luksan13ls(; n = 98, T = typeof(1.)) 
	return (x0 = -ones(T, n), obj = luksan13ls, grad! = luksan13ls_grad!)
end