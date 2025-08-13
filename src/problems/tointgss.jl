function tointgss(x, cache)
    c = 10 / (length(x) - 2)
	tointgssi(x0, x1, x2) = (c + x2^2) * (2 - exp(-(x0 - x1)^2 / (0.1 + x2^2)))
	return sumi(i -> tointgssi(x[i], x[i + 1], x[i + 2]), cache, dindices(x, 0, -2))
end

function tointgss_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    cst = 10 / (length(x) - 2)
    gradient[begin] = gradient[begin + 1] = 0
    @inbounds @simd for i in dindices(x, 0, -2)
        a = cst + x[i + 2]^2
        b = -(x[i] - x[i + 1])^2
        c_inv = 1 / (0.1 + x[i + 2]^2)
        d = - exp(b * c_inv)
        gradient[i] += a * (d * c_inv *(-2(x[i] - x[i + 1])))
        gradient[i + 1] += a * (d * c_inv *(2(x[i] - x[i + 1])))
        gradient[i + 2] = 2x[i + 2] * (2 + d) + a * (d * (-b * c_inv^2) * 2x[i + 2])
    end
    return gradient
end

"""
NAME          TOINTGSS

*   Problem :
*   *********

*   Toint's Gaussian problem.

*   This problem has N-2 trivial groups, all of which have 1 nonlinear
*   element

*   Source: problem 21 in
*   Ph.L. Toint,
*   "Test problems for partially separable optimization and results
*   for the routine PSPMIN",
*   Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

*   SIF input: Ph. Toint, Dec 1989, corrected Nick Gould, July 1993.

*   classification OUR2-AY-V-0

*IE N                   10             -PARAMETER     original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_tointgss(; n = 5000, T = typeof(1.))
	return (x0 = 3ones(T, n), obj = x -> tointgss(x, Vector{T}(undef, n)), grad! = tointgss_grad!)
end