function extpowell(x)
    s = zero(eltype(x))
    @simd for i in firstindices(x, length(x) ÷ 4)
        s += (x[4i - 3] + 10x[4i - 2])^2 + 5(x[4i - 1] - x[4i])^2 + 
             ((x[4i - 2] - 2x[4i - 1])^2)^2 + 10((x[4i - 3] - x[4i])^2)^2
    end
    return s / 2
end

function extpowell_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    @inbounds for i in firstindices(x, length(x) ÷ 4)
        xt1 = x[4i - 3] + 10x[4i - 2]
        xt2 = √5 * (x[4i - 1] - x[4i])
        xt3 = (x[4i - 2] - 2x[4i - 1])^2
        xt4 = √10 * (x[4i - 3] - x[4i])^2

        gradient[4i - 3] = xt1 + 2√10 * (x[4i - 3] - x[4i]) * xt4
        gradient[4i - 2] = 10xt1 + 2(x[4i - 2] - 2x[4i - 1]) * xt3
        gradient[4i - 1] = √5 * xt2 - 4(x[4i - 2] - 2x[4i - 1]) * xt3
        gradient[4i] = -√5 * xt2 - 2√10 * (x[4i-3] - x[4i]) * xt4
    end
    return gradient
end

"""
Problem 22 from Moré JJ, Garbow BS, Hillstrom KE: Testing unconstrained optimization software. ACM T Math Software. 1981

Keyword arguments:
; n = 100, T = typeof(1.)
"""
function gen_extpowell(; n = 100, T = typeof(1.))
    return (x0 = repeat(T[3., -1., 0., 1.]; inner = n ÷ 4), # Note, following optim here, but I think outer may have been intended?
            obj = extpowell, 
            grad! = extpowell_grad!)
end