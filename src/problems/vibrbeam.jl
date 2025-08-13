vibrbeami(x, p, v, a) = @inbounds((x[1] + p * (x[2] + p * (x[3] + p * x[4]))) *
	cos(x[5] + p * (x[6] + p * (x[7] + p * x[8])) - a) - v)^2

function vibrbeam(x, pos, vel, ang)
	check_x_indices(x, 8)
    return sumi(i -> @inbounds(vibrbeami(x, pos[i], vel[i], ang[i])), zero(eltype(x)), eachindex(pos))
end

function vibrbeam_grad!(gradient, x, pos, vel, ang)
	check_x_indices(x, 8)
    check_gradient_indices(gradient, x)
	g1 = g2 = g3 = g4 = g5 = g6 = g7 = g8 = zero(eltype(x))
	@inbounds @simd for i in eachindex(pos)
		p = pos[i]
		b = x[5] + p * (x[6] + p * (x[7] + p * x[8])) - ang[i]
		c = x[1] + p * (x[2] + p * (x[3] + p * x[4]))
		cosb = cos(b)
		a = 2(c * cosb - vel[i])
		gi = a * cosb
		g1 += gi
		g2 += gi * p
		g3 += gi * p^2
		g4 += gi * p^3
		gi = a * (-sin(b)) * c
		g5 += gi
		g6 += gi * p
		g7 += gi * p^2
		g8 += gi * p^3
	end
	gradient[1] = g1; gradient[2] = g2; gradient[3] = g3; gradient[4] = g4
	gradient[5] = g5; gradient[6] = g6; gradient[7] = g7; gradient[8] = g8
	return gradient
end

"""
NAME          VIBRBEAM

*   Problem:
*   ********

*   A nonlinear least-squares problem arising from laser-Doppler
*   measurements of a vibrating beam.  The data correspond to a simulated
*   experiment where two laser-Doppler velocimeters take measurements
*   at random points along the centreline of the beam.  These measurements
*   consist of a position (x), an incident angle (p) and the magnitude
*   of the velocity along the line of sight (v).
*   The problem is then to fit

*                         2      3                    2     3
*       v = (c + c x + c x  + c x ) cos[ d + d x + d x + d x  - p ]
*             0   1     2      3          0   1     2     3
*           <---- magnitude ----->       <------ phase ----->

*   in the least-squares sense.

*   Source: 
*   a modification of an exercize for L. Watson course on LANCELOT in
*   the Spring 1993. Compared to the original proposal, the unnecessary
*   elements were removed as well as an unnecessary constraint on the phase.

*   SIF input: Ph. L. Toint, May 1993, based on a proposal by
*              D. E. Montgomery, Virginia Tech., April 1993.

*   classification  SUR2-MN-8-0
...

Keyword argument:
; T = eltype(1.)
"""
function gen_vibrbeam(; T = eltype(1.))
    pos = T[39.1722, 53.9707, 47.9829, 12.5925, 16.5414, 18.9548, 27.7168, 31.9201, 45.683, 
    22.2524, 33.9805, 6.8425, 35.1677, 33.5682, 43.3659, 13.3835, 25.7273, 21.023, 10.9755, 
    1.5323, 45.4416, 14.5431, 22.4313, 29.0144, 25.2675, 15.5095, 9.6297, 8.3009, 30.8694, 
    43.3299]

    vel = T[-1.2026, 1.7053, 0.541, 1.1477, 1.2447, 0.9428, -0.136, -0.7542, -0.3396, 0.7057, 
    -0.8509, -0.1201, -1.2193, -1.0448, -0.7723, 0.4342, 0.1154, 0.2868, 0.3558, -0.509, 
    -0.0842, 0.6021, 0.1197, -0.1827, 0.1806, 0.5395, 0.2072, 0.1466, -0.2672, -0.3038]

    ang = T[2.5736, 2.7078, 2.6613, 2.0374, 2.1553, 2.2195, 2.4077, 2.4772, 2.6409, 2.2981, 
    2.5073, 1.838, 2.5236, 2.5015, 2.6186, 0.4947, 0.6062, 0.5588, 0.4772, 0.4184, 0.9051, 
    0.5035, 0.5723, 0.6437, 0.6013, 0.5111, 0.4679, 0.459, 0.6666, 0.863]

    return (x0 = T[-3.5, 1.0, 0.0, 0.0, 1.7, 0.0, 0.0, 0.0],
            obj = x -> vibrbeam(x, pos, vel, ang), 
            grad! = (gradient, x) -> vibrbeam_grad!(gradient, x, pos, vel, ang))
end