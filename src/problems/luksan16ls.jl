function luksan16ls(x, y)
    s = zero(eltype(x))
    for i in 0:2:96
		a = x[i + 1] + 2x[i + 2] + 3x[i + 3] + 4x[i + 4]
		for l in 1:4
			s += (y[l] - (exp(a / l) + 4exp(a / (2l)) + 9exp(a / (3l))) / l)^2
		end
    end
    return s
end

function luksan16ls_grad!(gradient, x, y)
    check_gradient_indices(gradient, x)
	gradient .= 0
    for i in 0:2:96, l in 1:4
		a = x[i + 1] + 2x[i + 2] + 3x[i + 3] + 4x[i + 4]
		b = y[l] - (exp(a / l) + 4exp(a / (2l)) + 9exp(a / (3l))) / l
		dsda = 2b * (-(exp(a / l) / l + 4exp(a / (2l)) / (2l) + 9exp(a / (3l)) / (3l)) / l)
		gradient[i + 1] += dsda
		gradient[i + 2] += 2dsda
		gradient[i + 3] += 3dsda
		gradient[i + 4] += 4dsda
    end
    return gradient
end

"""
NAME          LUKSAN16LS

*   Problem :
*   *********

*   Problem 16 (sparse exponential) in the paper

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
function gen_luksan16ls(; n = 100, T = typeof(1.)) 
    y = T[35.8, 11.2, 6.2, 4.4]
	return (x0 = repeat(T[-0.8,1.2,-1.2,0.8]; outer = n ÷ 4), 
			obj = x -> luksan16ls(x, y), 
			grad! = (gradient, x) -> luksan16ls_grad!(gradient, x, y))
end