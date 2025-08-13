dixon_price(x) = sumi(i -> i * (2x[i]^2 - x[i - 1])^2, (x[begin] - one(eltype(x)))^2, dindices(x, 1, 0))

function dixon_price_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    @inbounds begin
        gradient[begin] = 2(x[begin] - 1) - 4(2x[begin + 1]^2 - x[begin])
        for i in dindices(gradient, 1 , -1)
            gradient[i] = (8i) * (2x[i]^2 - x[i-1]) * x[i] - (2i + 2) * (2x[i + 1]^2 - x[i])
        end
        gradient[end] = (8lastindex(x)) * (2x[end]^2 - x[end-1]) * x[end]
    end
    return gradient
end

"""
Dixon and Price:
Problem 9 of Dixon, Price, 1989. The Truncated Newton Method for Sparse Unconstrained
Optimisation Using Automatic Differentiation. J. of optimization theory and applications, vol 60 no 2.
Note: I have not found a canonical starting point so I just specified one.

Keyword arguments:
; n = 100, T = typeof(1.)
"""
function gen_dixon_price(; n = 100, T = typeof(1.)) 
    return (x0 = ones(T, n), obj = dixon_price, grad! = dixon_price_grad!)
end