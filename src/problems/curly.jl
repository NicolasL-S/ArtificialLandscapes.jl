function curly(x, K)
    length(x) - K < 1 && throw(BoundsError("n should be greater than K."))
    T = eltype(x)
    s = q = zero(T)
	m = length(x) - K
	i = 1
    @inbounds while i <= m # We recompute the full sum only 1/11 times for precision. Otherwise, we just update it by removing the last term and adding the new one. Note, the double loop is needed to avoid using % (the mod operator) which is very slow.
        q = sumx(x, i:i + K)
        s += q * (q * (q^2 - 20) - T(0.1))
		i += 1
		if i <= m - 10
			for j in i:i+9
				q += - x[j - 1] + x[j + K]
				s += q * (q * (q^2 - 20) - T(0.1))
			end
			i += 10
		end
    end
    q = zero(T)
    @inbounds for i in lastindex(x):-1:lastindex(x) - K + 1
        q += x[i]
        s += q * (q * (q^2 - 20) - T(0.1))
    end
    return s 
end

@inline function add_arr!(a, c, range)
	@inbounds for i in range
		a[i] += c
	end
end

function curly_grad!(gradient,x, K)
    check_gradient_indices(gradient, x)
    T = eltype(x)
    length(x) - K < 1 && throw(ArgumentError("n should be greater than K."))
    q = zero(T)
    gradient .= 0
    @inbounds for i in dindices(gradient, 0, -K)
        q = i % 10 == 1 ? sumx(x, i:i + K) : q - x[i - 1] + x[i + K]
        a = 4q^3 - 40q - T(0.1)
        for j in i:i + K
            gradient[j] += a
        end
    end
    q = zero(T)
    @inbounds for i in lastindex(gradient):-1:lastindex(gradient) - K + 1
        q += x[i]
        a = 4q^3 - 40q - T(0.1)
        for j in i:lastindex(x)
            gradient[j] += a
        end
    end
    return gradient
end

"""
NAME          CURLY10

*   Problem :
*   --------

*   A banded function with semi-bandwidth 10 and
*   negative curvature near the starting point

*   Source: Nick Gould

*   SIF input: Nick Gould, September 1997.

*   classification OUR2-AN-V-0

*   Number of variables

*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   10000          -PARAMETER

####################################################################################################

NAME          CURLY20

*   Problem :
*   --------

*   A banded function with semi-bandwidth 20 and
*   negative curvature near the starting point

*   Source: Nick Gould

*   SIF input: Nick Gould, September 1997.

*   classification OUR2-AN-V-0

*   Number of variables

*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   10000          -PARAMETER
 
####################################################################################################

 NAME          CURLY30

*   Problem :
*   --------

*   A banded function with semi-bandwidth 30 and
*   negative curvature near the starting point

*   Source: Nick Gould

*   SIF input: Nick Gould, September 1997.

*   classification OUR2-AN-V-0

*   Number of variables

*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   10000          -PARAMETER
 ...

Keyword argument:
; n = 10000, K = 10, T = typeof(1.)
"""
function gen_curly(; n = 10000, K = 10, T = typeof(1.))
    return (x0 = T(0.0001) * collect(1:n) / (n + 1), 
            obj = x -> curly(x, K), 
            grad! = (gradient,x) -> curly_grad!(gradient,x, K))
end