function lanczosls(x, Y)
	s = zero(eltype(x))
	for i in eachindex(Y)
		s += (Y[i] - (x[1] * exp(-x[2] * 0.05(i-1)) + x[3] * exp(-x[4] * 0.05(i-1)) + 
			x[5] * exp(-x[6] * 0.05(i-1))))^2
	end
	return s
end

function lanczosls_grad!(gradient, x, Y)
    check_gradient_indices(gradient, x)
	g1 = g2 = g3 = g4 = g5 = g6 = zero(eltype(x))
	for i in eachindex(Y)
		a1 = exp(-x[2] * 0.05(i - 1))
		a2 = exp(-x[4] * 0.05(i - 1))
		a3 = exp(-x[6] * 0.05(i - 1))
		a = 2(Y[i] - (x[1] * a1 + x[3] * a2 + x[5] * a3))
		g1 += -a * a1
		g2 += a * x[1] * a1 * 0.05(i-1)
		g3 += -a * a2
		g4 += a * x[3] * a2 * 0.05(i-1)
		g5 += -a * a3
		g6 += a * x[5] * a3 * 0.05(i-1)
	end
	gradient[1] = g1; gradient[2] = g2; gradient[3] = g3
	gradient[4] = g4; gradient[5] = g5; gradient[6] = g6
	return gradient
end

"""
NAME          LANCZOS1LS

*   Problem :
*   *********

*   NIST Data fitting problem LANCZOS1.

*   Fit: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) + e

*   Source:  Problem from the NIST nonlinear regression test set
*     http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

*   Reference: Lanczos, C. (1956).
*     Applied Analysis. Englewood Cliffs, NJ:  Prentice Hall, pp. 272-280.

*   SIF input: Nick Gould and Tyrone Rees, Oct 2015

*   classification SUR2-MN-6-0
...

Keyword argument:
; T = typeof(1.)
"""
function gen_lanczos1ls(; T = typeof(1.))
	Y = T[ 2.5134, 2.044333373291, 1.668404436564, 1.366418021208, 
	1.123232487372, 0.9268897180037, 0.7679338563728, 0.6388775523106, 
	0.5337835317402, 0.4479363617347, 0.377584788435, 0.3197393199326, 
	0.2720130773746, 0.2324965529032, 0.1996589546065, 0.1722704126914, 
	0.1493405660168, 0.1300700206922, 0.1138119324644, 0.1000415587559, 
	0.0883320908454, 0.0783354401935, 0.06976693743449, 0.06239312536719]
    return (x0 = T[1.2, 0.3, 5.6, 5.5, 6.5, 7.6],
            obj = x -> lanczosls(x, Y),
            grad! = (gradient, x) -> lanczosls_grad!(gradient, x, Y))
end

"""
NAME          LANCZOS2LS

*   Problem :
*   *********

*   NIST Data fitting problem LANCZOS2.

*   Fit: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) + e

*   Source:  Problem from the NIST nonlinear regression test set
*     http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

*   Reference: Lanczos, C. (1956).
*     Applied Analysis. Englewood Cliffs, NJ:  Prentice Hall, pp. 272-280.

*   SIF input: Nick Gould and Tyrone Rees, Oct 2015

*   classification SUR2-MN-6-0
...

Keyword argument:
; T = typeof(1.)
"""
function gen_lanczos2ls(; T = typeof(1.))
	Y = T[2.5134, 2.04433, 1.6684, 1.36642, 1.12323, 0.92689, 0.767934, 
    0.638878, 0.533784, 0.447936, 0.377585, 0.319739, 0.272013, 0.232497, 
    0.199659, 0.17227, 0.149341, 0.13007, 0.113812, 0.100042, 0.0883321, 
    0.0783354,  0.0697669, 0.0623931]
    return (x0 = T[1.2, 0.3, 5.6, 5.5, 6.5, 7.6],
            obj = x -> lanczosls(x, Y),
            grad! = (gradient, x) -> lanczosls_grad!(gradient, x, Y))
end

"""
NAME          LANCZOS3LS

*   Problem :
*   *********

*   NIST Data fitting problem LANCZOS3.

*   Fit: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) + e

*   Source:  Problem from the NIST nonlinear regression test set
*     http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

*   Reference: Lanczos, C. (1956).
*     Applied Analysis. Englewood Cliffs, NJ:  Prentice Hall, pp. 272-280.

*   SIF input: Nick Gould and Tyrone Rees, Oct 2015

*   classification SUR2-MN-6-0
...

Keyword argument:
; T = typeof(1.)
"""
function gen_lanczos3ls(; T = typeof(1.))
	Y = T[2.5134, 2.0443, 1.6684, 1.3664, 1.1232, 0.9269, 0.7679, 0.6389, 
    0.5338, 0.4479, 0.3776, 0.3197, 0.272, 0.2325, 0.1997, 0.1723, 0.1493, 
    0.1301, 0.1138, 0.1, 0.0883, 0.0783, 0.0698, 0.0624]
    return (x0 = T[1.2, 0.3, 5.6, 5.5, 6.5, 7.6],
            obj = x -> lanczosls(x, Y),
            grad! = (gradient, x) -> lanczosls_grad!(gradient, x, Y))
end