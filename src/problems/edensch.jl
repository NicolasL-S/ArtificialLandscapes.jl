edensch(x) = sumi(i -> ((x[i] - 2)^2)^2 + ((x[i] - 2) * x[i + 1])^2 + (x[i + 1] + 1)^2, 
	eltype(x)(16), dindices(x, 0, -1))

function edensch_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	d = dl = zero(eltype(x))
    @inbounds for i in dindices(x, 0, -1)
        a = 4(x[i] - 2)^3
        b = 2(x[i] - 2) * x[i + 1]
        c = 2(x[i + 1] + 1)
		dl = d
		d = b * (x[i] - 2) + c
        gradient[i] = a + b * x[i + 1] + dl
    end
	gradient[end] = d
    return gradient
end

"""
NAME          EDENSCH

*   Problem :
*   *********

*   The extended Dennis and Schnabel problem, as defined by Li.

*   Source:
*   G. Li,
*   "The secant/finite difference algorithm for solving sparse
*   nonlinear systems of equations",
*   SIAM Journal on Optimization, (to appear), 1990.

*   SIF input: Ph. Toint, Apr 1990.
*              minor correction by Ph. Shott, January 1995.

*   classification OUR2-AN-V-0
...

Keyword arguments:
; n = 2000, T = typeof(1.)
"""
gen_edensch(; n = 2000, T = typeof(1.)) = (x0 = 8ones(T, n), obj = edensch, grad! = edensch_grad!)