rosenbrock(x) = (1 - x[1])^2 + 100(x[2] - x[1]^2)^2

function rosenbrock_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[1] = -2(1 - x[1]) - 400(x[2] - x[1]^2) * x[1]
    gradient[2] = 200(x[2] - x[1]^2)
    return gradient
end

"""
NAME          ROSENBR

*   Problem :
*   *********

*   The ever famous 2 variables Rosenbrock "banana valley" problem

*   Source:  problem 1 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-2-0
...

Keyword argument:
; T = typeof(1.)
"""
gen_rosenbrock(; T = typeof(1.)) = (x0 = T[-1.2, 1], obj = rosenbrock, grad! = rosenbrock_grad!)