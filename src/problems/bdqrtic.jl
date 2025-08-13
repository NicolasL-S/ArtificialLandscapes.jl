function bdqrtic(x, cache)
    check_x_indices(x, cache)
    return sumi(i -> (-4x[i] + 3)^2 + (x[i]^2 + 2x[i+1]^2 + 3x[i+2]^2 + 4x[i+3]^2 + 5x[end]^2)^2, 
        cache, dindices(x, 0, -4))
end

function bdqrtic_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    xN = x[end]
    gradient[begin] = gradient[begin + 1] = gradient[begin + 2] = gradient_N = zero(eltype(x))
    @inbounds @simd for i in dindices(gradient, 0, -4)
        a = 2(x[i]^2 + 2x[i+1]^2 + 3x[i+2]^2 + 4x[i+3]^2 + 5xN^2)
        gradient[i] += -8(-4x[i] + 3) + a * 2x[i]
        gradient[i + 1] += a * 4x[i+1]
        gradient[i + 2] += a * 6x[i+2]
        gradient[i + 3] = a * 8x[i+3]
        gradient_N += a * 10xN
    end
    gradient[end] = gradient_N
    return gradient
end

"""
NAME          BDQRTIC

*   Problem :
*   *********
*   This problem is quartic and has a banded Hessian with bandwidth = 9

*   Source: Problem 61 in
*   A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
*   "Performance of a multifrontal scheme for partially separable
*   optimization",
*   Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   Number of variables (variable)

*IE N                   100            -PARAMETER     original value
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
 ...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_bdqrtic(; n = 5000, T = typeof(1.))
    return (x0 = ones(T, n), obj = x -> bdqrtic(x, Vector{T}(undef,n)), grad! = bdqrtic_grad!)
end