penalty1(x) = sum(xi -> (xi - 1)^2, x) / 100000 + (sum(xi -> xi^2, x) - 0.25)^2

function penalty1_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    a = 4sum(xi -> xi^2, x) - 1
    @. gradient = (x - 1) / 50000 + a * x
end

"""
NAME          PENALTY1

*   Problem :
*   *********

*   This problem is a sum of n+1 least-squares groups, the first n of
*   which have only a linear element.
*   It Hessian matrix is dense.

*   Source:  Problem 23 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley #181 (p. 79).

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   N is the number of free variables

*IE N                   4              -PARAMETER
*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
 IE N                   1000           -PARAMETER
 ...

Keyword argument:
; n = 1000, T = typeof(1.)
"""
function gen_penalty1(; n = 1000, T = typeof(1.))
    return (x0 = T.(collect(1:n)), 
            obj = penalty1, 
            grad! = penalty1_grad!)
end