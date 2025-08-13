misra1als(x, Y, X) = sumi(i -> (Y[i] - x[1] * (1 - exp(-x[2] * X[i])))^2, zero(eltype(x)), eachindex(Y))

function misra1als_grad!(gradient, x, Y, X)
    check_x_indices(x, 2; regenerate = false)
    check_gradient_indices(gradient, x)
	g1 = g2 = zero(eltype(x))
	@inbounds @simd for i in eachindex(Y)
		e = exp(-x[2] * X[i])
		a = 2(Y[i] - x[1] * (1 - e))
		g1 += -a * (1 - e)
		g2 += -a * x[1] * e * X[i]
	end
	gradient[1] = g1; gradient[2] = g2
	return gradient
end

"""
NAME          MISRA1ALS

*   Problem :
*   *********

*   NIST Data fitting problem MISRA1A

*   Fit: y = b1*(1-exp[-b2*x]) + e

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
function gen_misra1als(; T = typeof(1.)) 

	X = T[77.6, 114.9, 141.1, 190.8, 239.9, 289, 332.8, 378.4, 434.8, 477.3, 
	536.8, 593.1, 689.1, 760]

	Y = [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 
	55.05, 61.01, 66.4, 75.47, 81.78]

	return (x0 = T[500,0.0001], 
			obj = x -> misra1als(x, Y, X), 
			grad! = (gradient, x) -> misra1als_grad!(gradient, x, Y, X))
end