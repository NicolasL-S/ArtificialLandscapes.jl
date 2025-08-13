in_perm_0dβ(x, i, β) = sumi(j -> (j + β) * (x[j]^i - 1 / j^i), zero(eltype(x)), eachindex(x))

perm_0dβ(x, β) = sumi(i -> in_perm_0dβ(x, i, β)^2, zero(eltype(x)), eachindex(x))

function perm_0dβ_grad!(gradient, x, β)
    check_gradient_indices(gradient, x)
    T = eltype(x)
	gradient .= 0
    for i in eachindex(x)
        a = 2sumi(j -> (j + β) * (x[j]^i - 1 / j^i), zero(T), eachindex(x))
        for j in eachindex(x)
            gradient[j] += a * (j + β) * (i * x[j]^(i - 1))
        end
    end
    return gradient
end

"""
Perm 0 d β
A. Hedar's test functions for unconstrained global optimization
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2545.htm
https://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.PermFunction02
Note: I could not find a cannonical starting point so I chose this one.

Keyword arguments:
; n = 4, T = typeof(1.), β = T(10)
"""
function gen_perm_0dβ(; n = 4, T = typeof(1.), β = T(10))
    return (x0 = -ones(T,n), 
            obj = x -> perm_0dβ(x, β), 
            grad! = (gradient, x) -> perm_0dβ_grad!(gradient, x, β))
end