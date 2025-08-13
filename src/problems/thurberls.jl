"""
NAME          THURBERLS 

*   Problem :
*   *********

*   NIST Data fitting problem THURBERLS.

*   Fit: y = (b1 + b2*x + b3*x**2 + b4*x**3) / 
*            (1 + b5*x + b6*x**2 + b7*x**3) + e

*   Source:  Problem from the NIST nonlinear regression test set
*     http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

*   Reference: Thurber, R., NIST (197?).  
*     Semiconductor electron mobility modeling.

*   SIF input: Nick Gould and Tyrone Rees, Oct 2015

*   classification SUR2-MN-7-0
...

Keyword argument:
; T = eltype(1.)
"""
function gen_thurberls(; T = eltype(1.))

	X = T[-3.067, -2.981, -2.921, -2.912, -2.84, -2.797, -2.702, -2.699, -2.633, -2.481, -2.363, 
	-2.322, -1.501, -1.46, -1.274, -1.212, -1.1, -1.046, -0.915, -0.714, -0.566, -0.545, -0.4, 
	-0.309, -0.109, -0.103, 0.01, 0.119, 0.377, 0.79, 0.963, 1.006, 1.115, 1.572, 1.841, 2.047, 
	2.2]

	Y = T[80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101, 92.405, 95.854, 100.696, 
	101.06, 401.672, 390.724, 567.534, 635.316, 733.054, 759.087, 894.206, 990.785, 1090.109, 
	1080.914, 1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509, 
	1425.208, 1421.384, 1442.962, 1464.35, 1468.705, 1447.894, 1457.628]

    return (x0 = T[ 1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03],
            obj = x -> hahn1ls(x, X, Y), 
            grad! = (gradient, x) -> hahn1ls_grad!(gradient, x, X, Y))
end