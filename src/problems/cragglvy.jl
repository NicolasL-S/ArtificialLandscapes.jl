function cragglvy(x, cache)
    check_x_indices(x, cache)
    return sumi(i -> @inbounds(((exp(x[2i - 1]) - x[2i])^2)^2 + 100((x[2i] - x[2i + 1])^2)^3 + 
    ((tan(x[2i + 1] - x[2i + 2]) + x[2i + 1] - x[2i + 2])^2)^2 + 
    ((x[2i - 1]^2)^2)^2 + (x[2i + 2] - 1)^2), cache, firstindices(x, length(x) ÷ 2 - 1))
end

function cragglvy_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[begin] = gradient[begin + 1] = 0
    for i in firstindices(x, length(gradient) ÷ 2 - 1)
        a1 = exp(x[2i - 1])
        a = 4(a1 - x[2i])^3
        @fastpow b = 600(x[2i] - x[2i + 1])^5
        c = 4(tan(x[2i + 1] - x[2i + 2]) + x[2i + 1] - x[2i + 2])^3 * 
            (sec(x[2i + 1] - x[2i + 2])^2 + 1)

        @fastpow gradient[2i - 1] += a * a1 + 8x[2i - 1]^7
        gradient[2i] += -a + b
        gradient[2i + 1] = -b + c
        gradient[2i + 2] = -c + 2x[2i + 2] - 2
    end
    gradient
end

"""
NAME          CRAGGLVY

*   Problem :
*   *********
*   Extended Cragg and Levy problem.
*   This problem is a sum of m  sets of 5 groups,
*   There are 2m+2 variables. The Hessian matrix is 7-diagonal.

*   Source:  problem 32 in
*   Ph. L. Toint,
*   "Test problems for partially separable optimization and results
*   for the routine PSPMIN",
*   Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

*   See  also Buckley#18
*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AY-V-0

*   M is the number of group sets

*IE M                   1              -PARAMETER n = 4     original value
*IE M                   4              -PARAMETER n = 10
*IE M                   24             -PARAMETER n = 50
*IE M                   49             -PARAMETER n = 100
*IE M                   249            -PARAMETER n = 500
*IE M                   499            -PARAMETER n = 1000
 IE M                   2499           -PARAMETER n = 5000

*   N is the number of variables
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_cragglvy(; n = 5000, T = typeof(1.))
    x0 = 2ones(T, n)
    x0[1] = 1
    return (x0 = x0, obj = x -> cragglvy(x, similar(x0)), grad! = cragglvy_grad!)
end