function chnrosnb(x, α²)
    check_x_indices(x, α²; regenerate = false)
    return sumi(i -> (x[i - 1] - x[i]^2)^2 * 16α²[i] +(x[i] - 1)^2, zero(eltype(x)), 
    dindices(x, 1, 0))
end

function chnrosnb_grad!(gradient, x, α²)
    check_x_indices(x, α²; regenerate = false)
    check_gradient_indices(gradient, x)
	b = bl = zero(eltype(x))
    @inbounds for i in dindices(x, 1, 0)
		bl = b
        a = 32(x[i - 1] - x[i]^2) * α²[i]
		b = 2x[i] * (1 - a) - 2
        gradient[i - 1] = a + bl
    end
	gradient[end] = b
    return gradient
end

"""
NAME          CHNROSNB

*   Problem :
*   --------
*   The chained Rosenbrock function (Toint)

*   Source:
*   Ph.L. Toint,
*   "Some numerical results using a sparse matrix updating formula in
*   unconstrained optimization",
*   Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

*   See also Buckley#46 (n = 25) (p. 45).
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   Number of variables ( at most 50)

*IE N                   10             -PARAMETER     original value
*IE N                   25             -PARAMETER
 IE N                   50             -PARAMETER
 ...

Keyword argument:
; T = typeof(1.)
"""
function gen_chnrosnb(; T = typeof(1.))

    α² = T[1.25, 1.4, 2.4, 1.4, 1.75, 1.2, 2.25, 1.2, 1, 1.1, 1.5, 1.6, 1.25, 1.25, 1.2, 1.2, 1.4, 
          0.5, 0.5, 1.25, 1.8, 0.75, 1.25, 1.4, 1.6, 2, 1, 1.6, 1.25, 2.75, 1.25, 1.25, 1.25, 3, 
          1.5, 2, 1.25, 1.4, 1.8, 1.5, 2.2, 1.4, 1.5, 1.25, 2, 1.5, 1.25, 1.4, 0.6, 1.5].^2

    return (x0 = -ones(T, 50), 
            obj = x -> chnrosnb(x, α²), 
            grad! = (gradient, x) -> chnrosnb_grad!(gradient, x, α²))
end