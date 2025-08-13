function liarwhd(x, cache)
    check_x_indices(x, cache)
    return 4sumi(i -> @inbounds(-x[begin] + x[i]^2)^2, cache, eachindex(x)) + 
	    sumi(i -> @inbounds(x[i] - 1)^2, cache, eachindex(x))
end

function liarwhd_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    @inbounds for i in eachindex(x)
        a = 8(x[i]^2 - x[begin])
        gradient[i] = (a + 1) * 2x[i] - 2
        gradient[begin] -= a
    end
    return gradient
end

"""
NAME          LIARWHD

*   Problem :
*   *********

*   Source:
*   G. Li,
*   "The secant/finite difference algorithm for solving sparse
*   nonlinear systems of equations",
*   SIAM Journal on Optimization, (to appear), 1990.

*   SIF input: Ph. Toint, Aug 1990.

*   classification SUR2-AN-V-0

*   This is a simplified version of problem NONDIA.

*   Number of variables (at least 2)

*IE N                   36             -PARAMETER     original value
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_liarwhd(; n = 5000, T = typeof(1.))
	return (x0 = 4ones(T, n), obj = x -> liarwhd(x, Vector{T}(undef, n)), grad! = liarwhd_grad!)
end