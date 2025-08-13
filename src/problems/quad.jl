function quad(x, vec, xt, mat, cache)
    check_x_indices(x, cache)
    return quadratic_form(mat, @.(xt = x - vec), cache) / 2
end

function quad_grad!(gradient, x, vec, xt, mat)
    check_x_indices(x, vec)
    check_gradient_indices(gradient, x)
    return mul!(gradient, mat, @.(xt = x - vec))
end

"""
Quadratic Diagonal
Problem C of Hans De Sterck. Steepest Descent Preconditioning for Nonlinear GMRES Optimization
(Problem B with a random non-diagonal matrix with condition number κ = n.)

Keyword argument:
; n = 100, T = typeof(1.)
"""
function gen_quad(; n = 100, T = typeof(1.))
    mat = sparse(T.(Diagonal(1:n)))
    x0 = zeros(T, n)
    vec = ones(T, n)
    xt = Vector{T}(undef,n)
    cache = Vector{T}(undef,n)
    obj = x -> quad(x, vec, xt, mat, cache)
    grad! = (gradient, x) -> quad_grad!(gradient, x, vec, xt, mat)
    return (x0 = x0, obj = obj, grad! = grad!)
end