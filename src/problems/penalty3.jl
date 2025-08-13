function penalty3(x)
    n = length(x)
	n > 2 || throw(BoundsError("x should have length at least 3"))
	T = eltype(x)
	T0, T1 = zero(T), one(T)
	r = sumi(i -> (x[i] + 2x[i + 1] + 10x[i + 2] - T1)^2, T0, dindices(x, 0,-2))
	s = sumi(i -> (2x[i] + x[i + 1] - 3T1)^2, T0, dindices(x, 0,-2))
    t = sum(abs2, x) - n^2
    u = sumi(i -> (x[i] - T1)^2, T0, firstindex(x):n ÷ 2)
    return T(0.001) * (r * exp(x[end]) + s * exp(x[end - 1]) + s * r) + t^2 + u
end

function penalty3_grad!(gradient, x, dr, ds)
    n = length(x)
	n > 2 || throw(BoundsError("x should have length at least 3"))
    check_gradient_indices(gradient, x)

    # In principle, those two shouldn't be necessary
    eachindex(x) == eachindex(dr) || throw(ArgumentError("x and dr should have the same indices"))
    eachindex(x) == eachindex(ds) || throw(ArgumentError("x and ds should have the same indices"))

	T = eltype(x)
    r = s = b = bl = a = al = al2 = zero(T)
    T1 = one(T)
    T1em3 = T(0.001)

    @inbounds for i in dindices(x,0,-2)
        al2 = al
        al = a
        a = (x[i] + 2x[i + 1] + 10x[i + 2] - T1)
        r += a^2
        dr[i] = 2a + 4al + 20al2

		bl = b
        b = (2x[i] + x[i + 1] - 3T1)
        s += b^2
        ds[i] = 4b + 2bl
    end
    dr[end - 1] = 4a + 20al
    dr[end] = 20a
	ds[end - 1] = 2b
	ds[end] = 0

    t = sum(abs2, x) - n^2

    expn = exp(x[end])
    expnm1 = exp(x[end-1])

    @. gradient = T1em3 * ((expn + s) * dr + (expnm1 + r) * ds) + 4t * x

    @inbounds for i in firstindices(x, n ÷ 2)
        gradient[i] += 2x[i] - 2T1
    end
    
    gradient[end - 1] += T1em3 * s * expnm1
    gradient[end] += T1em3 * r * expn
    return gradient
end

"""
NAME          PENALTY3

*   Problem :
*   *********

*   A penalty problem by Gill, Murray and Pitfield.
*   It has a dense Hessian matrix.

*   Source:  problem 114 (p. 81) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Nick Gould, Dec 1990.

*   classification OUR2-AY-V-0

*   N is the number of variables

*IE N/2                 25             -PARAMETER n = 50   original value
*IE N/2                 50             -PARAMETER n = 100
 IE N/2                 100            -PARAMETER n = 200
 ...

Keyword argument:
; n = 200, T = typeof(1.)
"""
function gen_penalty3(; n = 200, T = typeof(1.))
    x0 = T[-1 + 2(i % 2)  for i in 1:n]
    return (x0 = x0, 
            obj = penalty3, 
            grad! = (gradient, x) -> penalty3_grad!(gradient, x, similar(x0), similar(x0)))
end