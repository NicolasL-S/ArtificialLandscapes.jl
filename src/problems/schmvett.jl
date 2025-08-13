function schmvett(x)
    s = zero(eltype(x))
    @inbounds @simd for i in dindices(x, 0, -2)
        s += -(1 / (1 + (x[i] - x[i + 1])^2)) - sin(0.5(π * x[i+1] + x[i + 2])) -
            exp(-((x[i] + x[i + 2]) / x[i + 1] - 2)^2)
    end
    return s
end

function schmvett_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[1] = 0.
    gradient[2] = 0.
    @inbounds for i in dindices(x, 0, -2)
        da = 2(x[i] - x[i+1])/(1 + (x[i] - x[i+1])^2)^2
        db = -cos(0.5(π * x[i+1] + x[i+2]))
        xninv = 1 / x[i + 1]
        r = (x[i] + x[i + 2]) * xninv
        dc = 2exp(-(r - 2)^2) * (r - 2) * xninv
        gradient[i] += da + dc
        gradient[i + 1] += -da + 0.5π * db - r * dc
        gradient[i + 2] = 0.5db + dc
    end
    return gradient
end

"""
NAME          SCHMVETT

*   Problem :
*   *********

*   The Schmidt and Vetters problem.

*   This problem has N-2 trivial groups, all of which have 3 nonlinear
*   elements

*   Source:
*   J.W. Schmidt and K. Vetters,
*   "Albeitungsfreie Verfahren fur Nichtlineare Optimierungsproblem",
*   Numerische Mathematik 15:263-282, 1970.

*   See also Toint#35 and Buckley#14 (p90)

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AY-V-0

*   Number of variables

*IE N                   3              -PARAMETER     original value
*IE N                   10             -PARAMETER 
*IE N                   100            -PARAMETER 
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""

function gen_schmvett(; n = 5000, T = typeof(1.))
    return (x0 = 0.5ones(T, n), 
            obj = schmvett, 
            grad! = schmvett_grad!)
end