function tquartic(x, cache)
	check_x_indices(x, cache)
	return (x[begin] - 1)^2 + sumi(i -> (x[i]^2 - x[1]^2)^2, cache, dindices(x, 1,0))
end

function tquartic_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    x1_sq = x[begin]^2
    gr1 = 2x[begin] - 2
    @inbounds @simd for i in dindices(x, 1, 0)
        a = 4(x[i]^2 - x1_sq)
        gradient[i] = a * x[i]
        gr1 -= a * x[begin]
    end
    gradient[begin] = gr1
    return gradient
end

"""
NAME          TQUARTIC

*   Problem :
*   *********

*   A quartic function with nontrivial groups and
*   repetitious elements.

*   Source:
*   Ph. Toint, private communication.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   number of variables

*IE N                   5              -PARAMETER     original value
*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_tquartic(; n = 5000, T = typeof(1.))
	return (x0 = ones(T, n) / 10,
			obj = x -> tquartic(x, Vector{T}(undef, n)), 
			grad! = tquartic_grad!)
end