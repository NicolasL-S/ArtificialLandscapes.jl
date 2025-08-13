function nondquar(x)
	length(x)  < 2 && throw(BoundsError("x must have length at least 2"))
	return sumi(i -> ((x[i] + x[i + 1] + x[end])^2)^2, 
	(x[begin] - x[begin + 1])^2 + (x[end-1] - x[end])^2, dindices(x, 0, -2))
end

function nondquar_grad!(gradient, x)
	length(x)  < 2 && throw(BoundsError("x must have length at least 2"))
    check_gradient_indices(gradient, x)
	al = zero(eltype(x))
	@inbounds begin
		a = 2(x[begin] - x[begin + 1])
		gradient_end = -2(x[end - 1] - x[end])
		
		for i in dindices(x, 0, -2)
			al = a
			a = 4(x[i] + x[i + 1] + x[end])^3
			gradient_end += a
			gradient[i] = a + al
		end
		gradient[begin + 1] -= 2(x[begin] - x[begin + 1])
		gradient[end - 1] = 2(x[end - 1] - x[end]) + a
		gradient[end] = gradient_end
	end
    return gradient
end

"""
NAME          NONDQUAR

*   Problem :
*   *********

*   A nondiagonal quartic test problem.

*   This problem has an arrow-head type Hessian with a tridiagonal
*   central part and a border of width 1.
*   The Hessian is singular at the solution.

*   Source: problem 57 in
*   A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
*   "Performance of a multi-frontal scheme for partially separable
*   optimization"
*   Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

*   SIF input: Ph. Toint, Dec 1989.

*   classification OUR2-AN-V-0

*   Number of variables

*IE N                   100            -PARAMETER     original value
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_nondquar(;n = 5000, T = typeof(1.))
    return (x0 = T[-1 + 2(i % 2)  for i in 1:n], 
            obj = nondquar, 
            grad! = nondquar_grad!)
end