function luksan15ls(x, y)
    s = zero(eltype(x))
    for i in 0:2:96, l in 1:4
		a = abs(x[i + 1] * x[i + 2]^2 * x[i + 3]^3 * (x[i + 4]^2)^2)
		s += (y[l] - (a^(1 / l) + 4a^(1 / (2l)) + 9a^(1 / (3l))) / l)^2
    end
    return s
end

function luksan15ls_grad!(gradient, x, y)
    check_gradient_indices(gradient, x)
	gradient .= 0
    for i in 0:2:96, l in 1:4
		a = abs(x[i + 1] * x[i + 2]^2 * x[i + 3]^3 * (x[i + 4]^2)^2)
		b = (y[l] - (a^(1 / l) + 4a^(1 / (2l)) + 9a^(1 / (3l))) / l)
		dsda = -2b * ((1 / l) * a^(1 / l - 1) + (2 / l) * a^(1 / (2l)-1) + 
			(3 / l) * a^(1 / (3l) - 1)) / l * sign(x[i + 1]) * sign(x[i + 3])
		c = abs(x[i + 1]) 
		d = x[i + 2]^2
		e = abs(x[i + 3])^3
		f = (x[i + 4]^2)^2
		gradient[i + 1] += dsda *     d * e * f * sign(x[i + 1])
		gradient[i + 2] += dsda * c *     e * f * 2x[i + 2]
		gradient[i + 3] += dsda * c * d *     f * 3x[i + 3]^2 * sign(x[i + 3])
		gradient[i + 4] += dsda * c * d * e *     4x[i + 4]^3
    end
    return gradient
end

"""
NAME          LUKSAN15LS

*   Problem :
*   *********

*   Problem 15 (sparse signomial) in the paper

*     L. Luksan
*     Hybrid methods in large sparse nonlinear least squares
*     J. Optimization Theory & Applications 89(3) 575-595 (1996)

*   SIF input: Nick Gould, June 2017.

*   least-squares version

*   classification SUR2-AN-V-0
...

Note, the numbers below are the ones in the SIF file. But there was a mistake in the input in the 
SIF file from the original Problem 15 (sparse signomial):
     L. Luksan. Hybrid methods in large sparse nonlinear least squares. J. Optimization Theory & Applications 89(3) 575-595 (1996)

 - Parameters are [35.8, 11.2, 6.2, 4.4] instead of [14.4, 6.8, 4.2, 3.2] (they were probably switched with the numbers from LUKSAN16LS)
 - x^y = abs(x)^y instead of x^y = sign(x) * abs(x)^y

Keyword argument:
; n = 100, T = typeof(1.)
"""
function gen_luksan15ls(; n = 100, T = typeof(1.)) 
    y = T[35.8, 11.2, 6.2, 4.4] # Original numbers are [14.4, 6.8, 4.2, 3.2], they were probably switched them with the numbers from LUKSAN16LS
	return (x0 = repeat(T[-0.8,1.2,-1.2,0.8]; outer = n ÷ 4), 
			obj = x -> luksan15ls(x, y), 
			grad! = (gradient, x) -> luksan15ls_grad!(gradient, x, y))
end