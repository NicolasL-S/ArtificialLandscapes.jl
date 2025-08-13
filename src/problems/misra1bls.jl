misra1bls(x, Y, X) = sumi(i -> (Y[i] - x[1] * (1 - (1 + x[2] * X[i]/2)^(-2)))^2, zero(eltype(x)), eachindex(Y))

function misra1bls_grad!(gradient, x, Y, X)
    check_x_indices(x, 2; regenerate = false)
    check_gradient_indices(gradient, x)
	g1 = g2 = zero(eltype(x))
	@inbounds @simd for i in eachindex(Y)
		b = (1 - (1 + x[2] * X[i]/2)^(-2))
		a = 2(Y[i] - x[1] * b)
		g1 += -a * b
		g2 += -a * x[1] * 2(1 + x[2] * X[i]/2)^(-3) * X[i]/2
	end
	gradient[1] = g1; gradient[2] = g2
	return gradient
end

"""
NAME          MISRA1BLS

*   Problem :
*   *********

*   NIST Data fitting problem MISRA1B.

*   Fit: y = b1 * (1-(1+b2*x/2)**(-2)) + e

*   Source:  Problem from the NIST nonlinear regression test set
*     http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

*   Reference: Misra, D., NIST (1978).  
*     Dental Research Monomolecular Adsorption Study.

*   SIF input: Nick Gould and Tyrone Rees, Oct 2015

*   classification SUR2-MN-2-0
...

Keyword argument:
; T = typeof(1.)
"""
function gen_misra1bls(; T = typeof(1.)) 

	X = [77.6, 114.9, 141.1, 190.8, 239.9, 289, 332.8, 378.4, 434.8, 
	477.3, 536.8, 593.1, 689.1, 760]

	Y = [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 
	50.76, 55.05, 61.01, 66.4, 75.47, 81.78]

	return (x0 = T[500,0.0001], 
			obj = x -> misra1bls(x, Y, X), 
			grad! = (gradient, x) -> misra1bls_grad!(gradient, x, Y, X))
end