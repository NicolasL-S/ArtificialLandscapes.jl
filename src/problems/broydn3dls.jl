function broydn3dls(x, cache)
    check_x_indices(x, cache)
    b = ((3 - 2x[begin]) * x[begin] - 2x[begin + 1] + 1)^2
    m = sumi(i -> @inbounds((3 - 2x[i]) * x[i] - x[i - 1] - 2x[i + 1] + 1)^2, cache, dindices(x, 1,-1)) 
    e = ((3 - 2x[end]) * x[end] - x[end - 1] + 1)^2
    return b + m + e
end

function broydn3dls_grad!(gradient, x)
    length(x) >= 3 || throw(BoundsError("The length of x should be 3 or more."))
    check_gradient_indices(gradient, x)
    @inbounds begin
        b = 2((3 - 2x[begin]) * x[begin] - 2x[begin + 1] + 1)
        gradient[begin] = b * (3 - 4x[begin])
        gradient[begin + 1] = -2b
        e = 2((3 - 2x[end]) * x[end] - x[end - 1] + 1)
        for i in dindices(x, 1,-1)
            m = 2((3 - 2x[i]) * x[i] - x[i - 1] - 2x[i + 1] + 1)
            gradient[i-1] += -m
            gradient[i] += m * (3 - 4x[i])
            gradient[i+1] = -2m
        end
        gradient[end - 1] += -e
        gradient[end] += e * (3 - 4x[end])
    end
    return gradient
end

"""
NAME          BROYDN3DLS

*   Problem :
*   *********

*   Broyden tridiagonal system of nonlinear equations in the least
*   square sense.

*   Source:  problem 30 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Toint#17 and Buckley#78.
*   SIF input: Ph. Toint, Dec 1989.
*   Least-squares version: Nick Gould, Oct 2015.

*   classification SUR2-AN-V-0

*   N is the number of variables (variable).

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
function gen_broydn3dls(; n = 5000, T = typeof(1.))
    return (x0 = -ones(T, n),
            obj = x -> broydn3dls(x, Vector{T}(undef, n)), 
            grad! = broydn3dls_grad!)
end