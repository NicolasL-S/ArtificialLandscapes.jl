function ncb20(x, cache)
    check_x_indices(x, cache)
	n = length(x)
	n >= 30 || throw(ArgumentError("The length of x should be at least 30."))
	T = eltype(x)
	s = a = b = T0 = zero(T)
	cache .= x ./ (1 .+ x.^2)
	@inbounds for i in dindices(x,0,-30)
		if i % 20 == 1
			a = sumx(cache, i:i + 19)
			b = sumx(x, i:i + 19)
		else
			a += - cache[i - 1] + cache[i + 19]
			b += - x[i - 1] + x[i + 19]
		end
		s += 10a^2 / i - 2b / 10
	end
	s += sumi(i -> (x[i]^2)^2, cache, 1:n-10)
	s += sumi(i -> x[i] * x[i + 10] * x[i + n - 10] + 2x[i + n - 10]^2, T0, 1:10) / 10000
	return s + 2(n - 9)
end

function ncb20_grad!(gradient, x, cache)
    check_x_indices(x, cache)
	n = length(x)
	n >= 30 || throw(ArgumentError("The length of x should be at least 30."))
    check_gradient_indices(gradient, x)
	gradient .= 0
	T = eltype(x)
	cache .= one(T) ./ (1 .+ x.^2)
	
	a = c = T0 = zero(T)
	for i in dindices(x,0,-30)
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
	for i in 1:n-10
		gradient[i] += 4x[i]^3
	end
	for i in 1:10
		gradient[i] += x[i + 10] * x[i + n - 10]/10000
		gradient[i + 10] += x[i] * x[i + n - 10]/10000
		gradient[i + n - 10] += (x[i] * x[i + 10] + 4x[i + n - 10])/10000
	end
	return gradient
end

"""
NAME          NCB20

*   Problem :
*   *********

*   A banded problem with semi-bandwidth 20.  This problem exhibits frequent
*   negative curvature in the exact Hessian.

*   Source:
*   Ph. Toint, private communication, 1992.

*   SIF input: Ph. Toint, October 1992.

*   classification OUR2-AN-V-0

*   Problem dimension

*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   5000           -PARAMETER
 ...

Keyword argument:
; n = 5010, T = typeof(1.)
"""
function gen_ncb20(; n = 5010, T = typeof(1.))
	n >= 30 || throw(ArgumentError("The length of x should be at least 30."))
	x0 = zeros(T, n)
	x0[end - 9:end] .= 1
	cache = similar(x0)
    return (x0 = x0,
            obj = x -> ncb20(x, cache), 
            grad! = (gradient, x) -> ncb20_grad!(gradient, x, cache))
end