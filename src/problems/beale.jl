function beale_cst(x)
    T = eltype(x)
    return (T(1.5) - x[1] * (one(T) - x[2]), 
            T(2.25) - x[1] * (one(T) - x[2]^2), 
            T(2.625) - x[1] * (one(T) - x[2]^3))
end

beale(x) = sum(abs2, beale_cst(x))

function beale_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    T = eltype(x)
    c1, c2, c3 = beale_cst(x)
    gradient[1] = -2c1 * (1 - x[2]) - 2c2 * (1 - x[2]^2) + -2c3 * (1 - x[2]^3)
    gradient[2] =  2c1 * x[1] + 2c2 * 2x[1] * x[2] + 2c3 * 3x[1]*x[2]^2
    return gradient
end

"""
NAME          BEALE

*   Problem :
*   *********
*   Beale problem in 2 variables

*   Source: Problem 5 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#89.
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-2-0
...

Keyword argument:
; T = typeof(1.)
"""
gen_beale(; T = typeof(1.)) = (x0 = ones(T,2), obj = beale, grad! = beale_grad!)