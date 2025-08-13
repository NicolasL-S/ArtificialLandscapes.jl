function deconvu(x, TR)
    check_x_indices(x, 63; regenerate = false)
    l0, LGTR, LGSG = 12, 40, 11
    s = T0 = zero(eltype(x))
    for k in 1:LGTR
        s += (sumi(i -> @inbounds(x[l0 + LGTR + i] * x[l0 + k - i + 1]), 
			T0, 1:min(LGSG,k); simd = false) - TR[k])^2
    end
    return s
end

function deconvu_grad!(gradient, x, TR)
    check_x_indices(x, 63; regenerate = false)
    check_gradient_indices(gradient, x)
    l0, LGTR, LGSG = 12, 40, 11
    T0 = zero(eltype(x))
    gradient .= T0
    @inbounds for k in 1:LGTR
        a = 2(sumi(i -> @inbounds(x[l0 + LGTR + i] * x[l0 + k - i + 1]), 
			T0, 1:min(LGSG,k); simd = false) - TR[k])
        for i in 1:min(LGSG,k)
            gradient[l0 + LGTR + i] += a * x[l0 + k - i + 1]
            gradient[l0 + k - i + 1] += a * x[l0 + LGTR + i]
        end
    end
    return gradient
end

"""
NAME          DECONVU

*   Problem :
*   *********

*   A problem arising in deconvolution analysis 
*   (unconstrained version).

*   Source:  
*   J.P. Rasson, Private communication, 1996.

*   SIF input: Ph. Toint, Nov 1996.
*   unititialized variables fixed at zero, Nick Gould, Feb, 2013

*   classification SXR2-MN-61-0
...

Keyword arguments:
; T = typeof(1.)
"""
function gen_deconvu(; T = typeof(1.))
    TR = T[0, 0, 0.0016, 0.0054, 0.0702, 0.1876, 0.332, 0.764, 0.932, 0.812, 0.3464, 0.2064, 0.083, 
      0.034, 0.06179999, 1.2, 1.8, 2.4, 9, 2.4, 1.801, 1.325, 0.0762, 0.2104, 0.268, 0.552, 0.996, 
      0.36, 0.24, 0.151, 0.0248, 0.2432, 0.3602, 0.48, 1.8, 0.48, 0.36, 0.264, 0.006, 0.006]
    x0 = zeros(T,63)
    x0[53:63] .= T[0.01, 0.02, 0.4, 0.6, 0.8, 3, 0.8, 0.6, 0.44, 0.01, 0.01]

    return (x0 = x0,
            obj = x -> deconvu(x, TR),
            grad! = (gradient, x) -> deconvu_grad!(gradient, x, TR))
end