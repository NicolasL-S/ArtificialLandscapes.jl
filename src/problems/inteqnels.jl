function inteqnels(x, cache)
    check_x_indices(x, cache)
    h = 1 / eltype(x)(length(x) - 1)
    s = s1 = s2 = zero(eltype(x))
    for i in eachindex(x)
        cache[i] = (x[i] + h * (i - 1) + 1)^3
        s2 += (1 - h * (i - 1)) * cache[i]
    end
    @inbounds @simd for i in eachindex(x)
        s1 += h * (i - 1) * cache[i]
        s2 -= (1 - h * (i - 1)) * cache[i]
        s += (x[i] + 0.5h * ((1 - (i - 1) * h) * s1 + (i - 1) * h * s2))^2
    end
    return s
end

function inteqnels_grad!(gradient, x, cache, f)
    check_x_indices(x, cache)
    check_gradient_indices(gradient, x)
    h = 1 / eltype(x)(length(x) - 1)
    s1 = s2 = s3 = s4 = 0.
    @inbounds begin
        for i in eachindex(x)
            cache[i] = (x[i] + h * (i - 1) + 1)^3
            s2 += (1 - h * (i - 1)) * cache[i]
        end
        @simd for i in eachindex(x)
            s1 += h * (i - 1) * cache[i]
            s2 -= (1 - h * (i - 1)) * cache[i]
            f[i] = x[i] + 0.5h * ((1 - (i - 1) * h) * s1 + (i - 1) * h * s2)
            s4 += f[i] * (1 - (i - 1) * h)
        end
        @simd for i in eachindex(x)
            s3 += f[i] * (i - 1) * h
            s4 -= f[i] * (1 - (i - 1) * h)
            gradient[i] = 3h * ((1 - (i - 1) * h) * s3 + (i - 1) * h * s4) * 
				(x[i] + (i - 1) * h + 1)^2 + 2f[i]
        end
    end
    return gradient
end

"""
NAME          INTEQNELS

*   Problem :
*   *********
*   The discrete integral problem (INTEGREQ) without fixed variables
*   in least-squares form.

*   Source:  Problem 29 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   SIF input: Ph. Toint, Feb 1990.
*   Modification to remove fixed variables: Nick Gould, Oct 2015.

*   classification SUR2-AN-V-0

*   N+2 is the number of discretization points .
*   The number of free variables is N.

*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER     original value
*IE N                   100            -PARAMETER
 IE N                   500            -PARAMETER
 ...

Keyword argument:
; n = 500, T = typeof(1.)
"""
function gen_inteqnels(; n = 500, T = typeof(1.))
    x0 = zeros(T, n + 2)
    @inbounds for j in 1:n
        x0[j+1] = j/(n+1) * (j/(n+1) - 1)
    end
    cache = similar(x0)
    return (x0 = x0, 
            obj = x -> inteqnels(x, cache), 
            grad! = (gradient, x) -> inteqnels_grad!(gradient, x, cache, similar(x0)))
end