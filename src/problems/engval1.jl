function engval1(x, cache)
    check_x_indices(x, cache)
    return sumi(i -> @inbounds((x[i]^2 + x[i + 1]^2)^2 - 4x[i]), cache, dindices(x, 0, -1)) + 
        3length(x) - 3
end

function engval1_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    b = bl = zero(eltype(x))
    @inbounds for i in dindices(x, 0, -1)
		bl = b
        a = 4(x[i]^2 + x[i + 1]^2)
		b = a * x[i + 1]
        gradient[i] = a * x[i] - 4 + bl
    end
	gradient[end] = b
    return gradient
end

"""
NAME          ENGVAL1

*   Problem :
*   *********

*   The ENGVAL1 problem.
*   This problem is a sum of 2n-2 groups, n-1 of which contain 2 nonlinear
*   elements.

*   Source: problem 31 in
*   Ph.L. Toint,
*   "Test problems for partially separable optimization and results
*   for the routine PSPMIN",
*   Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

*   See also Buckley#172 (p. 52)
*   SIF input: Ph. Toint and N. Gould, Dec 1989.

*   classification OUR2-AN-V-0

*   N is the number of variables

*IE N                   2              -PARAMETER     original value
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
...

Keyword arguments:
; n = 5000, T = typeof(1.)
"""
function gen_engval1(; n = 5000, T = typeof(1.))
	return (x0 = 2ones(T, n), obj = x -> engval1(x, Vector{T}(undef, n)), grad! = engval1_grad!)
end