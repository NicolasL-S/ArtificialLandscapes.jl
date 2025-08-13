luksan11ls(x) = sumi(i -> (10(2x[i]/(1 + x[i]^2) - x[i+1]))^2 + (x[i] - 1)^2, zero(eltype(x)), 
	dindices(x, 0, -1))

function luksan11ls_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	a = al = zero(eltype(x))
    @inbounds for i in dindices(x, 0, -1)
        b = 1 / eltype(x)(1 + x[i]^2)
		al = a
        a = 200(2x[i] * b - x[i + 1])
        gradient[i] = 2a * (1 - x[i]^2) * b^2 + 2x[i] - 2 - al
    end
	gradient[end] = -a 
    return gradient
end

"""
NAME          LUKSAN11LS

*   Problem :
*   *********

*   Problem 11 (chained serpentine) in the paper

*     L. Luksan
*     Hybrid methods in large sparse nonlinear least squares
*     J. Optimization Theory & Applications 89(3) 575-595 (1996)

*   SIF input: Nick Gould, June 2017.

*   classification SUR2-AN-V-0
...

Keyword argument:
; n = 100, T = typeof(1.)
"""
function gen_luksan11ls(; n = 100, T = typeof(1.)) 
	return (x0 = -0.8ones(T, n), obj = luksan11ls, grad! = luksan11ls_grad!)
end