function paraboloid(x, vec, xt, mat, cache, α)
    check_x_indices(x, cache)
    xt1 = x[begin] - vec[begin]
    a = α * xt1^2
    @. xt = x - vec - a
    xt[begin] = xt1
    return quadratic_form(mat, xt, cache) / 2
end

function paraboloid_grad!(gradient, x, vec, xt, mat, α)
    check_x_indices(x, vec)
    check_gradient_indices(gradient, x)
    xt1 = x[begin] - vec[begin]
    a = α * xt1^2
    @. xt = x - vec - a
    xt[begin] = xt1
    mul!(gradient, mat, xt)
    gradient[begin] -= 2α * xt1 * (sum(gradient) - gradient[begin])
    return gradient
end

"""
Problem B from Hans De Sterck - Steepest descent preconditioning for nonlinear GMRES optimization

Keyword arguments:
; n = 100, T = typeof(1.)
"""
function gen_paraboloid_diagonal(; n = 100, T = typeof(1.))
    mat = T.(Diagonal(1:n))
    vec = ones(T, n)
    xt = Vector{T}(undef, n)
    cache = Vector{T}(undef, n)
    α = T(10)
    return (x0 = zeros(T, n),
            obj = x -> paraboloid(x, vec, xt, mat, cache, α),
            grad! = (gradient, x) -> paraboloid_grad!(gradient, x, vec, xt, mat, α))
end

"""
Problem B from Hans De Sterck - Steepest descent preconditioning for nonlinear GMRES optimization

Keyword arguments:
; n = 100, scaling = true, T = typeof(1.)
"""
function gen_paraboloid_random_matrix(; n = 100, scaling = true, T = typeof(1.))
    #init_stable_rand()
    #F = qr(stable_rand(n,n; D = Normal(), T))
    F = qr(randn(T, n,n))
    mat = scaling ? F.Q'*Diagonal(T.(float(1:n)))*F.Q : F.Q'F.Q
    vec = ones(T, n)
    xt = Vector{T}(undef, n)
    cache = Vector{T}(undef, n)
    α = T(10)
    return (x0 = zeros(n),
            obj = x -> paraboloid(x, vec, xt, mat, cache, α),
            grad! = (gradient, x) -> paraboloid_grad!(gradient, x, vec, xt, mat, α))
end