function powellsg(x)
	length(x) >= 4 || throw(BoundsError("x should have length at least 4"))
    s = zero(eltype(x))
    @inbounds @simd for i in firstindex(x):4:lastindex(x)
        s += (x[i] + 10x[i + 1])^2 + 5(x[i + 2] - x[i + 3])^2 + 
             ((x[i + 1] - 2x[i + 2])^2)^2 + 10((x[i] - x[i + 3])^2)^2
    end
    return s
end

function powellsg_grad!(gradient, x)
	length(x) >= 4 || throw(BoundsError("x should have length at least 4"))
    check_gradient_indices(gradient, x)
    @inbounds for i in firstindex(x):4:lastindex(x)
        a = 2(x[i] + 10x[i + 1])
        b = 10(x[i + 2] - x[i + 3])
        c = 4(x[i + 1] - 2x[i + 2])^3
        d = 40(x[i] - x[i + 3])^3

        gradient[i] = a + d
        gradient[i + 1] = 10a + c
        gradient[i + 2] = b - 2c
        gradient[i + 3] = -b -d
    end
    return gradient
end

"""
NAME          POWELLSG

*   Problem :
*   *********

*   The extended Powell singular problem.
*   This problem is a sum of n/4 sets of four terms, each of which is
*   assigned its own group.

*   Source:  Problem 13 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Toint#19, Buckley#34 (p.85)

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AN-V-0

*   N is the number of free variables, and should be a multiple of 4

*IE N                   4              -PARAMETER     original value
*IE N                   8              -PARAMETER
*IE N                   16             -PARAMETER
*IE N                   20             -PARAMETER
*IE N                   36             -PARAMETER
*IE N                   40             -PARAMETER
*IE N                   60             -PARAMETER
*IE N                   80             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_powellsg(;n = 5000, T = typeof(1.))
    x0 = repeat(T[3.,-1.,0.,1.]; outer = n ÷ 4)
    return (x0 = x0, 
            obj = powellsg, 
            grad! = powellsg_grad!)
end