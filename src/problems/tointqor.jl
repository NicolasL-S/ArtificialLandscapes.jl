function tointqor(x, v1)
	check_x_indices(x, 50)

    s = sumi(i -> x[i]^2 * v1[i], zero(eltype(x)), eachindex(x))
	T = eltype(x)
    @inbounds begin
        s += (-x[31] + x[1] + 5)^2 + T(1.5)*(-x[1] + x[2] + x[3] + 5)^2
        s += (-x[2] + x[4] + x[5] + 5)^2 + T(0.1)*(-x[4] + x[6] + x[7] + T(2.5))^2
        s += T(1.5) * (-x[6] + x[8] + x[9] + 6)^2 + 2(-x[8] + x[10] + x[11] + 6)^2
        s += (-x[10] + x[12] + x[13] + 5)^2 + T(1.5) * (-x[12] + x[14] + x[15] + 6)^2
        s += 3(- x[11] - x[13] - x[14] + x[16] + x[17] + 10)^2 + 2(-x[16] + x[18] + x[19] + 6)^2
        s += (-x[9] - x[18] + x[20] + 5)^2 + 3(-x[5] - x[20] - x[21] + 9)^2
        s += T(0.1) * (-x[19] + x[22] + x[23] + x[24] + 2)^2 + T(1.5) * (-x[23] + x[25] + x[26] + 7)^2
        s += T(0.15) * (-x[7] - x[25] + x[27] + x[28] + T(2.5))^2 + 2(-x[28] + x[29] + x[30] + 6)^2
        s += (-x[29] + x[31] + x[32] + 5)^2 + T(0.1) * (-x[32] + x[33] + x[34] + 2)^2
        s += 3(-x[3] - x[33] + x[35] + 9)^2 + T(0.1) * (-x[35] + x[21] + x[36] + 2)^2
        s += T(1.2) * (-x[36] + x[37] + x[38] + 5)^2 + (-x[30] - x[37] + x[39] + 5)^2
        s += T(0.1) * (-x[38] - x[39] + x[40] + T(2.5))^2 + 2(-x[40] + x[41] + x[42] + 5)^2
        s += T(1.2) * (-x[41] + x[43] + x[44] + x[50] + 6)^2 + 3(-x[44] + x[45] + x[46] + x[47] + 10)^2
        s += T(1.5) * (-x[46] + x[48] + 7)^2 + 3(-x[42] - x[45] - x[48] - x[50] + x[49] + 10)^2
        s += 2(-x[26] - x[34] - x[43] + 6)^2 + (-x[15] - x[17] - x[24] - x[47] + 5)^2
        s += T(1.2) * (-x[49] + 4)^2 + 2(-x[22] + 4)^2 + (-x[27] + 4)^2
    end
    return s
end

function tointqor_grad!(gradient, x, v1)
	check_x_indices(x, 50)
    check_gradient_indices(gradient, x)

	T = eltype(x)
    @. gradient = 2x * v1
    @inbounds begin
        w1 = 2(-x[31] + x[1] + 5)
        w2 = 3(-x[1] + x[2] + x[3] + 5)
        w3 = 2(-x[2] + x[4] + x[5] + 5)
        w4 = T(0.2) * (-x[4] + x[6] + x[7] + T(2.5))
        w5 = 3(-x[6] + x[8] + x[9] + 6)
        w6 = 4(-x[8] + x[10] + x[11] + 6)
        w7 = 2(-x[10] + x[12] + x[13] + 5)
        w8 = 3(-x[12] + x[14] + x[15] + 6)
        w9 = 6(-x[11] - x[13] - x[14] + x[16] + x[17] + 10)
        w10 = 4(-x[16] + x[18] + x[19] + 6)
        w11 = 2(-x[9] - x[18] + x[20] + 5)
        w12 = 6(-x[5] - x[20] - x[21] + 9)
        w13 = T(0.2) * (-x[19] + x[22] + x[23] + x[24] + 2)
        w14 = 3(-x[23] + x[25] + x[26] + 7)
        w15 = T(0.3) * (-x[7] - x[25] + x[27] + x[28] + T(2.5))
        w16 = 4(-x[28] + x[29] + x[30] + 6)
        w17 = 2(-x[29] + x[31] + x[32] + 5)
        w18 = T(0.2) * (-x[32] + x[33] + x[34] + 2)
        w19 = 6(-x[3] - x[33] + x[35] + 9)
        w20 = T(0.2) * (-x[35] + x[21] + x[36] + 2)
        w21 = T(2.4) * (-x[36] + x[37] + x[38] + 5)
        w22 = 2(-x[30] - x[37] + x[39] + 5)
        w23 = T(0.2) * (-x[38] - x[39] + x[40] + T(2.5))
        w24 = 4(-x[40] + x[41] + x[42] + 5)
        w25 = T(2.4) * (-x[41] + x[43] + x[44] + x[50] + 6)
        w26 = 6(-x[44] + x[45] + x[46] + x[47] + 10)
        w27 = 3(-x[46] + x[48] + 7)
        w28 = 6(-x[42] - x[45] - x[48] - x[50] + x[49] + 10)
        w29 = 4(-x[26] - x[34] - x[43] + 6)
        w30 = 2(-x[15] - x[17] - x[24] - x[47] + 5)
        gradient[1] += w1 - w2; gradient[2] += w2 - w3; gradient[3] += w2 - w19;
        gradient[4] += w3 - w4; gradient[5] += w3 - w12; gradient[6] += w4 - w5;
        gradient[7] += w4 - w15; gradient[8] += w5 - w6; gradient[9] += w5 - w11;
        gradient[10] += w6 - w7; gradient[11] += w6 - w9; gradient[12] += w7 - w8; 
        gradient[13] += w7 - w9; gradient[14] += w8 - w9; gradient[15] += w8 - w30;
        gradient[16] += w9 - w10; gradient[17] += w9 - w30; gradient[18] += w10 - w11; 
        gradient[19] += w10 - w13; gradient[20] += w11 - w12; gradient[21] += -w12 + w20;
        gradient[22] += w13 - 4(-x[22] + 4); gradient[23] += w13 - w14; 
        gradient[24] += w13 - w30; gradient[25] += w14; gradient[26] += w14 - w29;
        gradient[25] += -w15; gradient[27] += w15 - 2(-x[27] + 4);
        gradient[28] += w15 - w16; gradient[29] += w16 - w17; gradient[30] += w16 - w22;
        gradient[31] += -w1 + w17; gradient[32] += w17 - w18; gradient[33] += w18 - w19; 
        gradient[34] += w18 - w29; gradient[35] += w19 - w20; gradient[36] += w20 - w21;
        gradient[37] += w21 - w22; gradient[38] += w21 - w23; gradient[39] += w22 - w23;
        gradient[40] += w23; gradient[40] -= w24; gradient[41] += w24 - w25; 
        gradient[42] += w24 - w28; gradient[43] += w25 - w29; gradient[44] += w25 - w26; 
        gradient[45] += w26 - w28;  gradient[46] += w26 - w27; gradient[47] += w26 - w30;
        gradient[48] += w27 - w28; gradient[49] += w28 - T(2.4) * (-x[49] + 4); 
        gradient[50] += w25 - w28
    end
    return gradient
end

"""
NAME          TOINTQOR

*   Problem :
*   *********

*   Toint's  Quadratic Operations Research problem

*   Source:
*   Ph.L. Toint,
*   "Some numerical results using a sparse matrix updating formula in
*   unconstrained optimization",
*   Mathematics of Computation 32(1):839-852, 1978.

*   See also Buckley#55 (p.94) (With a slightly lower optimal value?)

*   SIF input: Ph. Toint, Dec 1989.

*   classification QUR2-MN-50-0

*   Number of variables

 IE N                   50
 ...

Keyword argument:
; n = 50, T = typeof(1.)
"""
function gen_tointqor(; n = 50, T = typeof(1.))

	v1 = T[1.25, 1.4, 2.4, 1.4, 1.75, 1.2, 2.25, 1.2, 1, 1.1, 1.5, 1.6, 1.25, 1.25, 1.2, 
	1.2,  1.4, 0.5, 0.5, 1.25, 1.8, 0.75, 1.25, 1.4, 1.6, 2, 1, 1.6, 1.25, 2.75, 1.25, 1.25, 
	1.25, 3, 1.5, 2, 1.25, 1.4, 1.8, 1.5, 2.2, 1.4, 1.5, 1.25, 2, 1.5, 1.25, 1.4, 0.6, 1.5]

	return (x0 = zeros(T, n), 
			obj = x -> tointqor(x,v1), 
			grad! = (gradient, x) -> tointqor_grad!(gradient, x, v1))
end