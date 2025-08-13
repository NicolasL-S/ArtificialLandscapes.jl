function srosenbr(x, cache)
	check_x_indices(x, cache)
	return sumi(i -> 100((x[2i] - x[2i-1]^2))^2 + (1 - x[2i-1])^2, cache, firstindex(x):lastindex(x) ÷ 2) 
end

function srosenbr_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    @inbounds for i in firstindex(x):lastindex(x) ÷ 2
        gradient[2i - 1] = -400x[2i - 1] * (x[2i] - x[2i - 1]^2) - 2(1 - x[2i - 1])
        gradient[2i] = 200(x[2i] - x[2i - 1]^2)
    end
    return gradient
end

"""
NAME          SROSENBR

*   Problem :
*   *********

*   The separable extension of Rosenbrock's function.

*   Source:  problem 21 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   N/2 is half the number of variables

*IE N/2                 5              -PARAMETER n = 10     original value
*IE N/2                 25             -PARAMETER n = 50
*IE N/2                 50             -PARAMETER n = 100
*IE N/2                 250            -PARAMETER n = 500
*IE N/2                 500            -PARAMETER n = 1000
 IE N/2                 2500           -PARAMETER n = 5000
*IE N/2                 5000           -PARAMETER n = 10000
...

Note: The starting point reproduces the output of the sif file: x0 = repeat(T[1.2, 1]; outer = n ÷ 2), even though the canonical starting point is -1.2, 1, -1.2, 1,...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_srosenbr(; n = 5000, T = typeof(1.))
	return (x0 = repeat(T[1.2,1]; outer = n ÷ 2),
			obj = x -> srosenbr(x, Vector{T}(undef, n)), 
			grad! = srosenbr_grad!)
end