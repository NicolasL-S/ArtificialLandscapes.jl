function ncb20b(x, cache)
    check_x_indices(x, cache)
	n = length(x)
	T = eltype(x)
	s = a = b = zero(T)
	cache .= x ./ (1 .+ x.^2)
	for i in dindices(x, 0, -19)
		if i % 20 == 1
			a = sumx(cache, i:i + 19)
			b = sumx(x, i:i + 19)
		else
			a += - cache[i - 1] + cache[i + 19]
			b += - x[i - 1] + x[i + 19]
		end
		s += 10a^2 / i -2b/10
	end
	s += 100sumi(i -> (x[i]^2)^2, cache, eachindex(x))
	return s + 2length(x)
end

function ncb20b_grad!(gradient, x, cache)
    check_x_indices(x, cache)
    check_gradient_indices(gradient, x)
	T = eltype(x)
	cache .= one(T) ./ (1 .+ x.^2)
	
	gradient .= 400x.^3
	a = c = T0 = zero(T)
	for i in dindices(x, 0, -19)
		if i % 20 == 1
			a = sumi(j -> x[i + j - 1] * cache[i + j - 1], T0, 1:20)
		else
			a += - x[i - 1] * cache[i - 1] + x[i + 19] * cache[i + 19]
		end
		c = 20 / i * a
		for j in 1:20
			gradient[i + j - 1] += c * ((1 - x[i + j - 1]^2) * cache[i + j - 1]^2) -0.2
		end
	end
	return gradient
end

"""
NAME          NCB20B

*   Problem :
*   *********

*   A banded problem with semi-bandwidth 20.  This problem exhibits frequent
*   negative curvature in the exact Hessian.  It is a simplified version of
*   problem NCB20.

*   Source:
*   Ph. Toint, private communication, 1993.

*   SIF input: Ph. Toint, April 1993.

*   classification OUR2-AN-V-0

*   Problem dimension

 IE N                   21             -PARAMETER     original value
*IE N                   22             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   180            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
*IE N                   2000           -PARAMETER
 IE N                   5000           -PARAMETER
 ...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_ncb20b(; n = 5000, T = typeof(1.))
	cache = Vector{T}(undef, n)
    return (x0 = zeros(T, n),
            obj = x -> ncb20b(x, cache), 
            grad! = (gradient, x) -> ncb20b_grad!(gradient, x, cache))
end