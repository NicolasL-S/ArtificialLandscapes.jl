eg2(x) = sumi(i -> sin(x[begin] + x[i]^2 - 1), zero(eltype(x)), dindices(x, 0, -1)) + 
	sin(x[end]^2) / 2

function eg2_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    s = zero(eltype(x))
    for i in dindices(x, 0, -1)
        a = cos(x[begin] + x[i]^2 - 1)
		s += a
        gradient[i] = a * 2x[i]
    end
    gradient[begin] += s
    gradient[end] = cos(x[end]^2) * x[end]
    return gradient
end

"""
NAME          EG2

*   Problem:
*   ********

*   A simple nonlinear problem given as an example in Section 1.2.4 of
*   the LANCELOT Manual.
*   The problem is non convex and has several local minima.

*   Source:
*   A.R. Conn, N. Gould and Ph.L. Toint,
*   "LANCELOT, A Fortran Package for Large-Scale Nonlinear Optimization
*   (Release A)"
*   Springer Verlag, 1992.

*   SIF input: N. Gould and Ph. Toint, June 1994.

*   classification OUR2-AN-1000-0
...

Keyword arguments:
; n = 1000, T = typeof(1.)
"""
gen_eg2(; n = 1000, T = typeof(1.)) = (x0 = zeros(T, n), obj = eg2, grad! = eg2_grad!)