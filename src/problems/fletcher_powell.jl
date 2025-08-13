θ(x) = x[1] > 0 ? atan(x[2] / x[1]) / (2π) : (π + atan(x[2] / x[1])) / (2π)

fletcher_powell(x) = 100((x[3] - 10θ(x))^2 + (sqrt(x[1]^2 + x[2]^2) - one(eltype(x)))^2) + x[3]^2

function fletcher_powell_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    sum_sq = x[1]^2 + x[2]^2
    T = eltype(x)
    if sum_sq == zero(T)
        dtdx1 = dtdx2 = zero(T)
    else
        a = one(T) / (2π * sum_sq)
        dtdx1 = - x[2] * a
        dtdx2 =   x[1] * a
    end

    s_sum_sq = sqrt(sum_sq)
    b = 200(s_sum_sq - one(T)) / s_sum_sq
    c = x[3] - 10θ(x)
    gradient[1] = -2000c * dtdx1 + b * x[1]
    gradient[2] = -2000c * dtdx2 + b * x[2]
    gradient[3] =  200c + 2x[3]
    return gradient
end

"""
Fletcher & Powell, 1963. A rapidly convergent descent method for minimization

Keyword arguments:
; T = typeof(1.)
"""
gen_fletcher_powell(; T = typeof(1.)) = (x0 = T[-1, 0, 0], obj = fletcher_powell, grad! = fletcher_powell_grad!)