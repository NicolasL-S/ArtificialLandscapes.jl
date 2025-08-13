function arwhead(x, cache)
    check_x_indices(x, cache)
    return 3length(x) - 3 + sumi(i -> -4x[i] + (x[i]^2 + x[end]^2)^2, cache, dindices(x,0,-1))
end

function arwhead_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    s = zero(eltype(x))
    @inbounds @simd for i in dindices(x,0,-1)
        a = 4(x[i]^2 + x[end]^2)
		s += a
        gradient[i] = -4 + a * x[i]
    end
    gradient[end] = s * x[end]
    return gradient
end

"""
NAME          ARWHEAD

*   Problem :
*   *********
*   A quartic problem whose Hessian is an arrow-head (downwards) with
*   diagonal central part and border-width of 1.

*   Source: Problem 55 in
*   A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
*   "Performance of a multifrontal scheme for partially separable
*   optimization",
*   Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AN-V-0

*   N is the number of variables

*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
 ...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_arwhead(; n = 5000, T = typeof(1.))
    return (x0 = ones(T, n), obj = x -> arwhead(x, Vector{T}(undef, n)), grad! = arwhead_grad!)
end