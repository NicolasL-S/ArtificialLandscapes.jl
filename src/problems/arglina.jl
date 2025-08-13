function arglina(x)
    n = eltype(x)(lastindex(x))
    a = sumx(x, eachindex(x))/n + 1
    return sumi(i -> (a - x[i])^2, n * a^2, eachindex(x))
end

arglina_grad!(gradient, x) = @. gradient = 2(x + 1)

"""
NAME          ARGLINA

*   Problem :
*   *********
*   Variable dimension full rank linear problem

*   Source: Problem 32 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Buckley#80 (with different N and M)
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
gen_arglina(; n = 200, T = typeof(1.)) = (x0 = ones(T, n), obj = arglina, grad! = arglina_grad!)