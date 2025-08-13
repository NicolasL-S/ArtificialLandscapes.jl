woodsi(x1, x2, x3, x4) = 100 * (x2 - x1^2)^2 + (1 - x1)^2 + 90 * (x4 - x3^2)^2 + (1 - x3)^2 + 
	10 * (x2 + x4 - 2)^2 + typeof(x1)(0.1) * (x2 - x4)^2

function woods(x)
	length(x) >= 4 || throw(BoundsError("x must have length at least 4."))
	r = firstindex(x):(firstindex(x) - 1 + length(x) ÷ 4)
	return sumi(i -> @inbounds(woodsi(x[4i - 3], x[4i - 2], x[4i - 1], x[4i])), zero(eltype(x)), r)
end

function woods_grad!(gradient, x)
	length(x) >= 4 || throw(BoundsError("x must have length at least 4."))
    check_gradient_indices(gradient, x)
	@inbounds for i in firstindex(x):(firstindex(x) - 1 + length(x) ÷ 4)
		j = 4i
	    a = 200 * (x[j - 2] - x[j - 3]^2)
		b = 2(1 - x[j - 3])
    	c = 180(x[j] - x[j - 1]^2)
		d = 2(1 - x[j - 1])
      	e = 20(x[j - 2] + x[j] - 2)
		f = eltype(x)(0.2) * (x[j - 2] - x[j])
		gradient[j] = c + e - f
		gradient[j - 1] = -2x[j - 1] * c - d
		gradient[j - 2] = a + e + f
		gradient[j - 3] = -2x[j - 3] * a - b
	end
	return gradient
end

"""
NAME          WOODS

*   Problem :
*   *********

*   The extended Woods problem.

*   This problem is a sum of n/4 sets of 6 terms, each of which is
*   assigned its own group.  For a given set i, the groups are
*   A(i), B(i), C(i), D(i), E(i) and F(i). Groups A(i) and C(i) contain 1
*   nonlinear element each, denoted Y(i) and Z(i).

*   The problem dimension is defined from the number of these sets.
*   The number of problem variables is then 4 times larger.

*   This version uses a slightly unorthodox expression of Woods
*   function as a sum of squares (see Buckley)

*   Source:  problem 14 in
*   J.J. More', B.S. Garbow and K.E. Hillstrom,
*   "Testing Unconstrained Optimization Software",
*   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

*   See also Toint#27, Buckley#17 (p. 101), Conn, Gould, Toint#7

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   NS is the number of sets (= n/4)

*IE NS                  1              -PARAMETER n= 4      original value
*IE NS                  25             -PARAMETER n = 100
*IE NS                  250            -PARAMETER n = 1000
 IE NS                  1000           -PARAMETER n = 4000
*IE NS                  2500           -PARAMETER n = 10000
...

Keyword argument:
; n = 4000, T = typeof(1.)
"""
function gen_woods(; n = 4000, T = typeof(1.))
	return (x0 = repeat(T[-3,-1], outer = n ÷ 2), obj = woods, grad! = woods_grad!)
end