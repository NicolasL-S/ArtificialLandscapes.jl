himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

function himmelblau_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    x1x2 = x[1] * x[2]
    gradient[1] = 4x[1]^3 + 4x1x2 - 42x[1] + 2x[2]^2 - 14
    gradient[2] = 2x[1]^2 - 22 + 4x1x2 + 4x[2]^3 - 26x[2]
end

"""
From Himmelblau, D.M. (1972), Applied Nonlinear Programming, McGraw-Hill, New York.

Keywords arguments:
; T = typeof(1.)
"""
gen_himmelblau(; T = typeof(1.)) = (x0 = T[2, 2], obj = himmelblau, grad! = himmelblau_grad!)