function dqdrtic(x, cache)
    check_x_indices(x, cache)
    return x[begin]^2 + 101x[begin + 1]^2 + 200x[end - 1]^2 + 100x[end]^2 + 
        sumi(i -> @inbounds(201x[i]^2), cache, dindices(x, 2, -2))
end

function dqdrtic_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	
    gradient[begin] = 2x[begin]
    gradient[begin + 1] = 202x[begin + 1]
    @inbounds for i in dindices(x, 2, -2)
        gradient[i] = 402x[i]
    end
    gradient[end - 1] = 400x[end - 1]
    gradient[end] = 200x[end]
    return gradient
end

"""
NAME          DQDRTIC

*   Problem :
*   *********

*   A simple diagonal quadratic.

*   Source: problem 22 in
*   Ph. L. Toint,
*   "Test problems for partially separable optimization and results
*   for the routine PSPMIN",
*   Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

*   SIF input: Ph. Toint, Dec 1989.

*   classification QUR2-AN-V-0

*   N is the number of variables (variable)

*IE N                   10             -PARAMETER     original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
...

Keyword arguments:
; n = 5000, T = typeof(1.)
"""
function gen_dqdrtic(; n = 5000, T = typeof(1.))
    return (x0 = 3ones(T, n), obj = x -> dqdrtic(x, Vector{T}(undef, n)), grad! = dqdrtic_grad!)
end