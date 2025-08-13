function trigonometric(x, cosx)
    check_x_indices(x, cosx)
    cosx .= cos.(x)
    scos = sumx(cosx)
	T0 = zero(eltype(x))
	l = eltype(x)(length(x))
    return sumi(i -> (l + i * (1 - cosx[i]) - sin(x[i]) - scos)^2, T0, eachindex(x)) / 2
end

function trigonometric_grad!(gradient, x, cosx, sinx)
    check_x_indices(x, cosx)
    check_gradient_indices(gradient, x)
    n = lastindex(x)
    cosx .= cos.(x)
    sinx .= sin.(x)
    scos = sum(cosx)
    @. gradient = n + (1:n) * (1 - cosx) - sinx - scos

    sxt = sum(gradient)
    @. gradient = sxt*sin(x) + gradient * ((1:n) * sinx - cosx)
end

"""
Problem 26 from Moré JJ, Garbow BS, Hillstrom KE: Testing unconstrained optimization software. ACM T Math Software. 1981

*Note: this one is almost the same as CUTEst's TRIGON1 which has sin(x[i]) in the parenthesis

Keyword arguments:
; n = 100, T = typeof(1.)
"""
function gen_trigonometric(; n = 100, T = typeof(1.))
    x0 = ones(T, n)/n
    cosx = similar(x0)
    sinx = similar(x0)
    return (x0 = x0, 
    		obj = x -> trigonometric(x, cosx), 
    		grad! = (gradient, x) -> trigonometric_grad!(gradient, x, cosx, sinx))
end