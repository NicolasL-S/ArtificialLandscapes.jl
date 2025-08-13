function quartc(x, cache)
	check_x_indices(x, cache)
	return sumi(i -> ((x[i] - i)^2)^2, cache, eachindex(x))
end

function quartc_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    @inbounds @simd for i in eachindex(gradient)
        gradient[i] = 4(x[i] - i)^3
    end
    return gradient
end

"""
NAME          QUARTC

*   Problem :
*   *********

*   A simple quartic function.

*   Source:  problem 157 (p. 87) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, March 1991.

*   classification OUR2-AN-V-0

*   number of variables

*IE N                   25             -PARAMETER     original value
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_quartc(;n = 5000, T = typeof(1.))
    return (x0 = 2ones(T, n), 
            obj = x -> quartc(x, Vector{T}(undef, n)), 
            grad! = quartc_grad!)
end