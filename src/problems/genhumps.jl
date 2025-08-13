function genhumps(x, ζ, cache)
    check_x_indices(x, cache)
	T005 = eltype(x)(0.05)
    return sumi(i -> @inbounds(sin(ζ * x[i])^2 * sin(ζ * x[i + 1])^2 +  
		T005 * (x[i]^2 + x[i + 1]^2)), cache, dindices(x, 0, -1))
end

function genhumps_grad!(gradient, x, ζ, sinζx, cosζx) # Without the shortcut with cos = sqrt(1-sin²)
    check_gradient_indices(gradient, x)
	T = eltype(x)
	T01 = T(0.1)
	@inbounds for i in eachindex(x)
		cosζx[i] = cos(ζ * x[i])
		sinζx[i] = sin(ζ * x[i])
	end
	c = cl = zero(T)
    @inbounds for i in dindices(x, 0, -1)
        a = sinζx[i]
        a_sq = a^2
        b = sinζx[i + 1]
        b_sq = b^2
		cl = c
		c = 2ζ * a_sq * b * cosζx[i + 1] + T01 * x[i+1]
        gradient[i] = 2ζ * a * cosζx[i] * b_sq + T01 * x[i] + cl
    end
	gradient[end] = c
    return gradient
end

"""
NAME          GENHUMPS

*   Problem :
*   *********

*   A multi-dimensional variant of HUMPS, a two dimensional function
*   with a lot of humps. The density of humps increases with the
*   parameter ZETA, making the problem more difficult.

*   The problem is nonconvex.

*   Source:
*   Ph. Toint, private communication, 1997.

*   SDIF input: N. Gould and Ph. Toint, November 1997.

*   classification OUR2-AN-V-0

*   Number of variables

*IE N                   5              -PARAMETER
*IE N                   10             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER    original value
 IE N                   5000           -PARAMETER
...

Keyword arguments
; n = 5000, density = 20, T = typeof(1.)
"""
function gen_genhumps(; n = 5000, density = 20, T = typeof(1.)) # also possible density = 2.
    x0 = T(-506.2)*ones(T, n)
    x0[1] = -506
	cache1 = similar(x0)
	cache2 = similar(x0)
    return (x0 = x0, 
            obj = x -> genhumps(x, T.(density), cache1), 
            grad! = (x, gradient) -> genhumps_grad!(x, gradient, T.(density), cache1, cache2))
end