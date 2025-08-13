function biggsi(x, ci, i, T0) # I don't know why but adding this T0 variable seems to make the function faster.
	a = eltype(x)(-0.1)
	return (ci + x[3] * exp(a * i * x[1]) - x[4] * exp(a * i * x[2]) + x[6] * exp(a * i * x[5]))^2
end

function biggs6(x, c)
    check_x_indices(x, 6; regenerate = false)
	T0 = zero(eltype(x))
	return sumi(i -> (biggsi(x, c[i], i, T0)), T0, eachindex(c))
end

function biggs6_grad!(gradient, x, c)
    check_x_indices(x, 6; regenerate = false)
    check_gradient_indices(gradient, x)
	g1 = g2 = g3 = g4 = g5 = g6 = zero(eltype(x))
	a = eltype(x)(-0.1)
	for i in eachindex(c)
		a1 = exp(a * i * x[1])
		a2 = exp(a * i * x[2])
		a3 = exp(a * i * x[5])
		b = 2(c[i] + x[3] * a1 - x[4] * a2 + x[6] * a3)
		g1 += b * x[3] * a1 * a * i
		g2 += -b * x[4] * a2 * a * i
		g3 += b * a1
		g4 += -b * a2
		g5 += b * x[6] * a3 * a * i
		g6 += b * a3
	end
	gradient[1] = g1; gradient[2] = g2; gradient[3] = g3
	gradient[4] = g4; gradient[5] = g5; gradient[6] = g6
	return gradient
end

"""
NAME          BIGGS6

*   Problem :
*   *********
*   Biggs EXP problem in 6 variables

*   Source: Problem 21 in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-6-0
 ...

Keyword argument:
; T = typeof(1.)
"""
function gen_biggs6(; T = typeof(1.))
	c = T[-exp(-0.1*i) + 5exp(-i) - 3exp(-0.4*i) for i in 1:13]
	x0 = ones(T, 6)
	x0[2] = 2
    return (x0 = x0,
            obj = x -> biggs6(x, c),
            grad! = (gradient, x) -> biggs6_grad!(gradient, x, c))
end