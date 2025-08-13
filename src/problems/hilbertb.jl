function hilbertb(x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        for j in 1:i-1
            s += x[i] * x[j] / (i + j - 1)
        end
        s += x[i]^2 * (5 + 1 / (4i - 2))
    end
    return s
end

function hilbertb_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	gradient .= 0
    @inbounds for i in eachindex(x)
        gradient[i] = x[i] * (10 + 2 / eltype(x)(4i - 2))
        for j in 1:i-1
            a = eltype(x)(1 / (i + j - 1))
            gradient[i] += a * x[j] 
            gradient[j] += a * x[i] 
        end
    end
    return gradient
end

"""
NAME          HILBERTB

*   Problem :
*   *********

*   The perturbed Hilbert quadratic

*   Source: problem 19 (p. 59) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification QUR2-AN-V-0

*   Dimension of the problem

*IE N                   5              -PARAMETER     original value
 IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
...

Keyword argument:
; n = 10, T = typeof(1.)
"""
gen_hilbertb(; n = 10, T = typeof(1.)) = (x0 = -3ones(T, n), obj = hilbertb, grad! = hilbertb_grad!)