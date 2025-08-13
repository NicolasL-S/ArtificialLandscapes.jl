function hosaki(x)
    T = eltype(x)
    x1_sq = x[1]^2
    return (one(T) - 8x[1] + 7x1_sq - (7 / 3) * x[1]^3 + x1_sq^2 / 4) * x[2]^2 * exp(-x[2])
end

function hosaki_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    T = eltype(x)
    x1² = x[1]^2
    x1⁴div4 = x1²^2 / 4
    exp_mx2 = exp(-x[2])
    gradient[1] = (x[1]^3 - 7x[1]^2 + 14x[1] - 8)* x[2]^2 * exp_mx2
    gradient[2] = 2(one(T) - 8x[1] + 7x1² - (7 / 3) * x[1]^3 + x1⁴div4) * x[2] * exp_mx2 - 
                 (one(T) - 8x[1] + 7x1² - (7 / 3) * x[1]^3 + x1⁴div4) * x[2]^2 * exp_mx2
    return gradient
end

"""
Originally from 
Uosaki, H. Imamura, M. Tasaka, H. Sugiyama. 1970. A heuristic method for maxima searching in case of 
multimodal surfaces. Technology Reports of the Osaka University 

Keyword arguments:
; T = typeof(1.)
"""
gen_hosaki(; T = typeof(1.)) = (x0 = T[3.6, 1.9], obj = hosaki, grad! = hosaki_grad!)

# add constraints: lower [0.,0.], upper [5.,6.]