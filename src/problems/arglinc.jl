function arglinc(x)
    s_in = sumi(i -> i * x[i], zero(eltype(x)), dindices(x, 1, -1))
    rangei = firstindex(x) + 1:firstindex(x) + 2length(x) - 2
    return 2 + sumi(i -> ((i - 1) * s_in - 1)^2, zero(eltype(x)), rangei)
end

function arglinc_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    s_in = sumi(i -> i * x[i], zero(eltype(x)), dindices(x, 1, -1))
    rangei = firstindex(x) + 1:firstindex(x) + 2length(x) - 2
    ds_out = 2sumi(i -> (i - 1) * ((i - 1) * s_in - 1), zero(eltype(x)), rangei)
    gradient[begin] = 0
    for i in dindices(x, 1, -1)
        gradient[i] = i * ds_out
    end
    gradient[end] = 0
    return gradient
end

"""
NAME          ARGLINC

*   Problem :
*   *********
*   Variable dimension rank one linear problem, with zero rows and columns

*   Source: Problem 34 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#101 (with different N and M)
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   This problem is a linear least squares

*   N is the number of free variables
*   M is the number of equations ( M.ge.N)

*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER 
*IE N                   100            -PARAMETER
 IE N                   200            -PARAMETER

*IE M                   20             -PARAMETER .ge. N
*IE M                   100            -PARAMETER .ge. N
*IE M                   200            -PARAMETER .ge. N
 IE M                   400            -PARAMETER .ge. N
 ...

Keyword argument:
; n = 200, T = typeof(1.)
"""
gen_arglinc(; n = 200, T = typeof(1.)) = (x0 = ones(T, n), obj = arglinc, grad! = arglinc_grad!)