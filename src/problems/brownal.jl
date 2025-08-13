function brownal(x)
    length(x) >= 10 || throw(BoundsError("Length of x should be at least 10."))
    a = sum(x) - length(x) - 1
    p = one(eltype(x))
    for i in firstindices(x, 10)
        p *= x[i]
    end
    return sumi(i -> (x[i] + a)^2, zero(eltype(x)), dindices(x,0,-1)) + (p - 1)^2
end

function brownal_grad!(gradient, x)
    length(x) >= 10 || throw(BoundsError("The length of x should be 10 or more."))
    check_gradient_indices(gradient, x)
    T = eltype(x)
    n = length(x)

    s = sum(x)
    a = n * (s - T(n + 1)) + s - x[end]
    p = one(T)
    for i in firstindices(x, 10)
        p *= x[i]
    end

    for i in firstindices(x, 10)
        gradient[i] = 2(x[i] + a + (p - 1) * p / x[i])
    end
    for i in dindices(x, 10, -1)
        gradient[i] = 2(x[i] + a)
    end
    gradient[end] = 2(n * (s - n) + one(T) - x[end])
    n == 10 && (gradient[end] += 2(p - 1) * p / x[end])

    return gradient
end

"""
NAME          BROWNAL

*   Problem :
*   *********
*   Brown almost linear least squares problem.
*   This problem is a sum of n least-squares groups, the last one of
*   which has a nonlinear element.
*   It Hessian matrix is dense.

*   Source: Problem 27 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#79
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   N is the number of free variables (variable).

*IE N                   10             -PARAMETER     original value
*IE N                   100            -PARAMETER
 IE N                   200            -PARAMETER
*IE N                   1000           -PARAMETER
...

Note: contrary to the Brown almost linear (27) function in Moré, Garbow and Hillsttom, CUTEst only 
uses the first 10 elements for the product. The same is done here.

Keyword argument:
; n = 200, T = typeof(1.)
"""
gen_brownal(; n = 200, T = typeof(1.)) = (x0 = ones(T, n) ./ 2, obj = brownal, grad! = brownal_grad!)