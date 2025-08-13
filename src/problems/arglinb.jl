function arglinb(x)
    s_in = sumi(i -> i * x[i], zero(eltype(x)), eachindex(x))
    return sumi(i -> (i * s_in - 1)^2, zero(eltype(x)), 1:2length(x))
end

function arglinb_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    s_in = sumi(i -> i * x[i], zero(eltype(x)), eachindex(x))
    ds_out = 2sumi(i -> i * (i * s_in - 1), zero(eltype(x)), 1:2length(x))
    for i in eachindex(gradient)
        gradient[i] = i * ds_out
    end
    return gradient
end

"""
NAME          ARGLINB

*   Problem :
*   *********
*   Variable dimension rank one linear problem

*   Source: Problem 33 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#93 (with different N and M)
*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   This problem is a linear least squares

*   N is the number of free variables
*   M is the number of equations ( M .ge. N)

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
gen_arglinb(; n = 200, T = typeof(1.)) = (x0 = ones(T, n), obj = arglinb, grad! = arglinb_grad!)