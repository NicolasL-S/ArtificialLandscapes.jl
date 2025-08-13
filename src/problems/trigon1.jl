function trigon1(x, cosx)
	check_x_indices(x, cosx)
    cosx .= cos.(x)
    scos = sumx(cosx)
	T0 = zero(eltype(x))
	l = eltype(x)(length(x))
    return sumi(i -> (l - scos + i * (1 - cosx[i] - sin(x[i])))^2, T0, eachindex(x))
end

function trigon1_grad!(gradient, x, cosx, sinx)
	check_x_indices(x, cosx)
    check_gradient_indices(gradient, x)
    n = length(x)
    cosx .= cos.(x)
    sinx .= sin.(x)
    scos = sum(cosx)
    @. gradient = n + (1:n) * (1 - cosx - sinx) - scos

    sxt = sum(gradient)
    @. gradient = sxt*sin(x) + gradient * ((1:n) * (sinx - cosx))
	@. gradient *= 2
	return gradient
end

"""
NAME          TRIGON1

*   Problem :
*   *********

*   SCIPY global optimization benchmark example Trigonometric01

*   Fit: y = sum_{j=1}^{n} cos(x_j) + i (cos(x_i) + sin(x_i) ) + e

*   Source:  Problem from the SCIPY benchmark set
*     https://github.com/scipy/scipy/tree/master/benchmarks/ ...
*             benchmarks/go_benchmark_functions

*   SIF input: Nick Gould, Jan 2020

*   classification SUR2-MN-V-0

*   Number of variables

 IE N                   10             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER
...

Keyword argument:
; n = 10, T = typeof(1.)
"""
function gen_trigon1(; n = 10, T = typeof(1.))
	cosx = Vector{T}(undef, n)
	sinx = Vector{T}(undef, n)
	return (x0 = ones(T, n) / 10,
			obj = x -> trigon1(x, cosx), 
			grad! = (gradient, x) -> trigon1_grad!(gradient, x, cosx, sinx))
end