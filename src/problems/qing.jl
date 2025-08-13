qing(x) = sumi(i -> (x[i]^2 - i)^2, zero(eltype(x)), eachindex(x))

function qing_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    @inbounds @simd for i in eachindex(gradient)
		gradient[i] = 4(x[i]^2 - i) * x[i]
    end
    return gradient
end

"""
NAME          QING

*   Problem :
*   *********

*   SCIPY global optimization benchmark example Qing

*   Fit: y  = x_i^2 + e

*   Source:  Problem from the SCIPY benchmark set
*     https://github.com/scipy/scipy/tree/master/benchmarks/ ...
*             benchmarks/go_benchmark_functions

*   SIF input: Nick Gould, Jan 2020

*   classification SUR2-MN-V-0

*   Number of variables

 IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 100, T = typeof(1.)
"""
function gen_qing(; n = 100, T = typeof(1.))
    return (x0 = ones(T, n), 
            obj = qing, 
            grad! = qing_grad!)
end