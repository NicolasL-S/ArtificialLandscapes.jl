large_polynomial(x) =  sumi(i -> (i - x[i])^2, zero(eltype(x)), eachindex(x))

function large_polynomial_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    for i in eachindex(x)
        gradient[i] = -2(i - x[i])
    end
end

"""
#= Large-Scale Quadratic
From OptimTestProblems:
https://github.com/JuliaNLSolvers/OptimTestProblems.jl/blob/c1fba66c90b44934d13cd11b2502573f1c11fab8/src/optim_tests/multivariate/from_optim.jl#L224

Keyword arguments:
; n = 250, T = typeof(1.)
"""
function gen_large_polynomial(; n = 250, T = typeof(1.))
    return (x0 = zeros(T, n), obj = large_polynomial, grad! = large_polynomial_grad!)
end