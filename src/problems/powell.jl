powell(x) = (x[1] + 10x[2])^2 + 5(x[3] - x[4])^2 + ((x[2] - 2x[3])^2)^2 + 10((x[1] - x[4])^2)^2

function powell_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    a³ = (x[1] - x[4])^3
    b³ = (x[2] - 2x[3])^3
    gradient[1] = 2(x[1] + 10x[2]) + 40a³
    gradient[2] = 20(x[1] + 10x[2]) + 4b³
    gradient[3] = 10(x[3] - x[4]) - 8b³
    gradient[4] = -10(x[3] - x[4]) - 40a³
    return gradient
end

"""
Powell’s Quadratic Problem

Problem 35 in Ali, Khompatraporn, & Zabinsky: A Numerical Evaluation of Several Stochastic 
Algorithms on Selected Continuous Global Optimization Test
www.researchgate.net/profile/Montaz_Ali/publication/226654862_A_Numerical_Evaluation_of_Several_Stochastic_Algorithms_on_Selected_Continuous_Global_Optimization_Test_Problems/links/00b4952bef133a1a6b000000.pdf
Originally from Wolfe, M.A. (1978), Numerical Methods for Unconstrained Optimization, Van Nostrand
Reinhold Company, New York.

Keyword arguments:
; T = typeof(1.)
"""
gen_powell(; T = typeof(1.)) = (x0 = T[3, -1, 0, 1], obj = powell, grad! = powell_grad!)

# Note: should add constraints: lower = [-10,-10,-10,-10], upper = [10, 10, 10, 10]