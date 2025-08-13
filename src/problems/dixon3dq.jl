function dixon3dq(x, cache)
    check_x_indices(x, cache)
    return (x[begin] - 1)^2 + (x[end] - 1)^2 + sumi(i -> @inbounds(x[i] - x[i + 1])^2, cache, 
        dindices(x, 1, -1))
end

function dixon3dq_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[begin] = 2x[begin] - 2
	a = al = zero(eltype(x))
    @inbounds for i in dindices(x, 1, -1)
		al = a
        a = 2(x[i] - x[i+1])
        gradient[i] = a - al
    end
    gradient[end] = 2x[end] - 2 - a
    return gradient
end

"""
NAME          DIXON3DQ

*   Problem :
*   *********

*   Dixon's tridiagonal quadratic.

*   Source: problem 156 (p. 51) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification QUR2-AN-V-0

*   Number of variables (variable)

*IE N                   10             -PARAMETER     original value
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   10000          -PARAMETER
...

Keyword arguments:
; n = 10_000, T = typeof(1.)
"""

function gen_dixon3dq(; n = 10_000, T = typeof(1.))
    return (x0 = -ones(T, n),
            obj = x -> dixon3dq(x, Vector{T}(undef, n)),
            grad! = dixon3dq_grad!)
end