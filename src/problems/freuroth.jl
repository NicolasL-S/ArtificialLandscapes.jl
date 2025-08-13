function freuroth(x, cache)
    check_x_indices(x, cache)
    return sumi(i -> ((5 - x[i + 1]) * x[i + 1]^2 + x[i] - 2x[i + 1] - 13)^2 +
	    ((1 + x[i + 1]) * x[i + 1]^2 + x[i] - 14x[i + 1] - 29)^2, cache, dindices(x,0,-1))
end

function freuroth_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[begin] = 0
    @inbounds @simd for i in dindices(x, 0, -1)
        a = 2((5 - x[i + 1]) * x[i + 1]^2 + x[i] - 2x[i + 1] - 13)
        b = 2((1 + x[i + 1]) * x[i + 1]^2 + x[i] - 14x[i + 1] - 29)
        gradient[i] += a + b
        gradient[i + 1] = (10a + 2b) * x[i + 1] + (-3a + 3b) * x[i + 1]^2 - 2a - 14b
    end
    return gradient
end

"""
NAME          FREUROTH

*   Problem :
*   *********

*   The Freudentstein and Roth test problem

*   Source: problem 2 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Toint#33, Buckley#24
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   N is the number of variables

*IE N                   2              -PARAMETER     original value
*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
...

Keyword arguments:
; n = 5000, T = typeof(1.)
"""
function gen_freuroth(; n = 5000, T = typeof(1.))
    x0 = zeros(T, n)
    x0[1] = 1/2
    x0[2] = -2
    return (x0 = x0, obj = x -> freuroth(x, similar(x0)), grad! = freuroth_grad!)
end