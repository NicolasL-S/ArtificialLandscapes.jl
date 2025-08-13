extrosnb(x) = sumi(i -> 100(x[i] - x[i-1]^2)^2, (x[begin] - 1)^2, dindices(x, 1, 0))

function extrosnb_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[begin] = 2x[begin] - 2
    @inbounds @simd for i in dindices(x, 1, 0)
        gradient[i] = 200(x[i] - x[i-1]^2)
        gradient[i - 1] += -gradient[i] * 2x[i-1]
    end
    return gradient
end

"""
NAME          EXTROSNB

*   Problem :
*   --------

*   The extended Rosenbrock function (nonseparable version).

*   Source: problem 10 in
*   Ph.L. Toint,
*   "Test problems for partially separable optimization and results
*   for the routine PSPMIN",
*   Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

*   See also Buckley#116.  Note that MGH#21 is the separable version.
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   Number of variables

 IE N                   5              -PARAMETER     original value
*IE N                   10             -PARAMETER
*IE N                   100            -PARAMETER
 IE N                   1000           -PARAMETER
...

Keyword arguments:
; n = 1000, T = typeof(1.)
"""
gen_extrosnb(; n = 1000, T = typeof(1.)) = (x0 = -ones(T, n), obj = extrosnb, grad! = extrosnb_grad!)