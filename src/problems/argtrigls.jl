function argtrigls(x, cosx, sinx)
    check_x_indices(x, cosx)
    cosx .= cos.(x)
    sinx .= sin.(x)
    scosx = sum(cosx)
    return sumi(i -> (i * @inbounds(cosx[i] + sinx[i]) + scosx - (length(x) + i))^2, 
        zero(eltype(x)), eachindex(x))
end

function argtrigls_grad!(gradient, x, cosx, sinx, cache)
    check_x_indices(x, cosx)
    check_gradient_indices(gradient, x)
    n = length(x)
    cosx .= cos.(x)
    sinx .= sin.(x)
    scosx = sum(cosx)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        cache[i] = 2(i * @inbounds(cosx[i] + sinx[i]) + scosx - (n + i))
        s += cache[i]
    end
    @inbounds for i in eachindex(x)
        gradient[i] = cache[i] * i * (-sinx[i] + cosx[i]) - sinx[i] * s
    end
    return gradient
end

"""
NAME          ARGTRIGLS

*   Problem :
*   *********

*   Variable dimension trigonometric problem in least-squares form.
*   This problem is a sum of n least-squares groups, each of
*   which has n+1 nonlinear elements.  Its Hessian matrix is dense.

*   Source:  Problem 26 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   SIF input: Ph. Toint, Dec 1989.
*   Least-squares version: Nick Gould, Oct 2015.

*   classification SUR2-AN-V-0

*   N is the number of free variables

*IE N                   10             -PARAMETER original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
 IE N                   200            -PARAMETER
 ...

Keyword argument:
; n = 200, T = typeof(1.)
"""
function gen_argtrigls(; n = 200, T = typeof(1.))
    cosx = Vector{T}(undef, n)
    sinx = Vector{T}(undef, n)
    return (x0 = ones(T, n)/n, 
            obj = x -> argtrigls(x, cosx, sinx), 
            grad! = (gradient, x) -> argtrigls_grad!(gradient, x, cosx, sinx, Vector{T}(undef, n)))
end