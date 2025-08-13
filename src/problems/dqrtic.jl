function dqrtic(x, cache)
    check_x_indices(x, cache)
    return sumi(i -> ((x[i] - i)^2)^2, cache, eachindex(x))
end

function dqrtic_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    for i in eachindex(gradient)
        gradient[i] = 4(x[i] - i)^3
    end
    return gradient
end

"""
NAME          DQRTIC

*   Problem :
*   *********
*   Variable dimension diagonal quartic problem.

*   Source: problem 157 (p. 87) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AN-V-0

*   Number of variables (variable)

*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER     original value
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
...

Keyword arguments:
; n = 5000, T = typeof(1.)
"""
function gen_dqrtic(; n = 5000, T = typeof(1.))
    return (x0 = 2ones(T, n), obj = x -> dqrtic(x, Vector{T}(undef, n)), grad! = dqrtic_grad!)
end