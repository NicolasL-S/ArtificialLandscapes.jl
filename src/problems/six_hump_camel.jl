six_hump_camel(x) = (4 - eltype(x)(2.1) * x[1]^2 + eltype(x)(1 / 3) * (x[1]^2)^2) * x[1]^2 + 
    x[1] * x[2] + (-4 + 4x[2]^2) * x[2]^2

function six_hump_camel_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    x1² = x[1]^2
	T = eltype(x)
    gradient[1] = (T(-4.2) * x[1] +  (4 / 3) * x[1]^3) * x1² + 
        2(4 - T(2.1) * x1² +  T(1 / 3) * x1²^2) * x[1] + x[2] 
    gradient[2] = x[1] + (8x[2]) * x[2]^2 + 2(-4 + 4x[2]^2) * x[2]
    return gradient
end

"""
# Six-hump camel
# https://infinity77.net/global_optimization/test_functions_nd_S.html#go_benchmark.SixHumpCamel
# Note: I have not found a canonical starting point so I just specified one

Keyword argument:
; T = typeof(1.)
"""
function gen_six_hump_camel(; T = typeof(1.))
    return (x0 = T[1, 1], obj = six_hump_camel, grad! = six_hump_camel_grad!)
end