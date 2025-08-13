function power(x, cache)
	check_x_indices(x, cache)
	return sumi(i -> i * x[i]^2, cache, eachindex(x))^2
end

function power_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    s = zero(eltype(x))
    @inbounds @simd for i in eachindex(gradient)
        a = eltype(x)(i) * x[i]
        s += a * x[i]
        gradient[i] = a
    end
    gradient .*= 4s
    return gradient
end

"""
NAME          POWER

*   Problem :
*   *********

*   The Power problem by Oren.

*   Source:
*   S.S. Oren,
*   Self-scaling variable metric algorithms,
*   Part II: implementation and experiments"
*   Management Science 20(5):863-874, 1974.

*   See also Buckley#179 (p. 83)

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AN-V-0

*   Number of variables

*IE N                   10             -PARAMETER     original value
*IE N                   20             -PARAMETER
*IE N                   30             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   75             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
*IE N                   5000           -PARAMETER
 IE N                   10000          -PARAMETER
 ...

Keyword argument:
; n = 10000, T = typeof(1.)
"""
function gen_power(; n = 10000, T = typeof(1.))
    return (x0 = ones(T, n), 
            obj = x -> power(x, Vector{T}(undef, n)), 
            grad! = power_grad!)
end