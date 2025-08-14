const landscapes = Dict{AbstractString, Function}()

include("problems/dixon_price.jl")
landscapes["Dixon and Price"] = gen_dixon_price

include("problems/extpowell.jl")
landscapes["Extended Powell"] = gen_extpowell

include("problems/fletcher_powell.jl")
landscapes["Fletcher-Powell"] = gen_fletcher_powell

include("problems/himmelblau.jl")
landscapes["Himmelblau"] = gen_himmelblau

include("problems/hosaki.jl")
landscapes["Hosaki"] = gen_hosaki

include("problems/large_polynomial.jl")
landscapes["Large-Scale Quadratic"] = gen_large_polynomial

include("problems/paraboloid.jl")
landscapes["Paraboloid Diagonal"] = gen_paraboloid_diagonal
# landscapes["Paraboloid Random Matrix"] = gen_paraboloid_random_matrix # Best not to include it since it is random

include("problems/perm_2.jl")
landscapes["Perm 2"] = gen_perm_0dβ

include("problems/powell.jl")
landscapes["Powell"] = gen_powell

include("problems/quad.jl")
landscapes["Quadratic Diagonal"] = gen_quad

include("problems/six_hump_camel.jl")
landscapes["Six-hump camel"] = gen_six_hump_camel

include("problems/trigonometric.jl")
landscapes["Trigonometric"] = gen_trigonometric

# From CUTEst
include("problems/arglina.jl")
landscapes["ARGLINA"] = gen_arglina

include("problems/arglinb.jl")
landscapes["ARGLINB"] = gen_arglinb

include("problems/arglinc.jl")
landscapes["ARGLINC"] = gen_arglinc

include("problems/argtrigls.jl")
landscapes["ARGTRIGLS"] = gen_argtrigls

include("problems/arwhead.jl")
landscapes["ARWHEAD"] = gen_arwhead

include("problems/bdqrtic.jl")
landscapes["BDQRTIC"] = gen_bdqrtic

include("problems/beale.jl")
landscapes["BEALE"] = gen_beale

include("problems/biggs6.jl")
landscapes["BIGGS6"] = gen_biggs6

include("problems/brownal.jl")
landscapes["BROWNAL"] = gen_brownal

include("problems/broydn3dls.jl")
landscapes["BROYDN3DLS"] = gen_broydn3dls

include("problems/chnrosnb.jl")
landscapes["CHNROSNB"] = gen_chnrosnb

include("problems/cliff.jl")
landscapes["CLIFF"] = gen_cliff

include("problems/cosine.jl")
landscapes["COSINE"] = gen_cosine

include("problems/cragglvy.jl")
landscapes["CRAGGLVY"] = gen_cragglvy

include("problems/curly.jl")
landscapes["CURLY10"] = (; n = 10000, T = eltype(1.)) -> gen_curly(;n, K = 10, T)
landscapes["CURLY20"] = (; n = 10000, T = eltype(1.)) -> gen_curly(;n, K = 20, T)
landscapes["CURLY30"] = (; n = 10000, T = eltype(1.)) -> gen_curly(;n, K = 30, T)

include("problems/deconvu.jl")
landscapes["DECONVU"] = gen_deconvu

include("problems/dixmaan.jl")
landscapes["DIXMAANA1"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0), T(0.125), T(0.125), (0,0,0,0);n, T)
landscapes["DIXMAANB"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.0625), T(0.0625), T(0.0625), (0,0,0,0); n, T)
landscapes["DIXMAANC"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.125), T(0.125), T(0.125), (0,0,0,0); n, T)
landscapes["DIXMAAND"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.26), T(0.26), T(0.26), (0,0,0,0); n, T)
landscapes["DIXMAANE1"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0), T(0.125), T(0.125), (1,0,0,1);n, T)
landscapes["DIXMAANF"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.0625), T(0.0625), T(0.0625), (1,0,0,1); n, T)
landscapes["DIXMAANG"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.125), T(0.125), T(0.125), (1,0,0,1); n, T)
landscapes["DIXMAANH"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.26), T(0.26), T(0.26), (1,0,0,1); n, T)
landscapes["DIXMAANI1"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0), T(0.125), T(0.125), (2,0,0,2);n, T)
landscapes["DIXMAANJ"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.0625), T(0.0625), T(0.0625), (2,0,0,2); n, T)
landscapes["DIXMAANK"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.125), T(0.125), T(0.125), (2,0,0,2); n, T)
landscapes["DIXMAANL"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.26), T(0.26), T(0.26), (2,0,0,2); n, T)
landscapes["DIXMAANM1"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0), T(0.125), T(0.125), (2,1,1,2);n, T)
landscapes["DIXMAANN"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.0625), T(0.0625), T(0.0625), (2,1,1,2); n, T)
landscapes["DIXMAANO"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.125), T(0.125), T(0.125), (2,1,1,2); n, T)
landscapes["DIXMAANP"] = (; n = 3000, T = eltype(1.)) -> 
    gen_dixmaan(T(1), T(0.26), T(0.26), T(0.26), (2,1,1,2); n, T)

include("problems/dixon3dq.jl")
landscapes["DIXON3DQ"] = gen_dixon3dq

include("problems/dqdrtic.jl")
landscapes["DQDRTIC"] = gen_dqdrtic

include("problems/dqrtic.jl")
landscapes["DQRTIC"] = gen_dqrtic

include("problems/edensch.jl")
landscapes["EDENSCH"] = gen_edensch

include("problems/eg2.jl")
landscapes["EG2"] = gen_eg2

include("problems/engval1.jl")
landscapes["ENGVAL1"] = gen_engval1

include("problems/extrosnb.jl")
landscapes["EXTROSNB"] = gen_extrosnb

include("problems/fletchcr.jl")
landscapes["FLETCHCR"] = gen_fletchcr

include("problems/fminsrf2.jl")
include("problems/fminsurf.jl") # fminsurf needs fminsrf2 to be included before
landscapes["FMINSRF2"] = gen_fminsrf2
landscapes["FMINSURF"] = gen_fminsurf

include("problems/freuroth.jl")
landscapes["FREUROTH"] = gen_freuroth

include("problems/gaussls.jl")
landscapes["GAUSS1LS"] = gen_gauss1ls
landscapes["GAUSS2LS"] = gen_gauss2ls
landscapes["GAUSS3LS"] = gen_gauss3ls

include("problems/genhumps.jl")
landscapes["GENHUMPS"] = gen_genhumps

include("problems/genrose.jl")
landscapes["GENROSE"] = gen_genrose

include("problems/hahn1ls.jl")
landscapes["HAHN1LS"] = gen_hahn1ls

include("problems/hilbertb.jl")
landscapes["HILBERTB"] = gen_hilbertb

include("problems/indef.jl")
landscapes["INDEF"] = gen_indef

include("problems/inteqnels.jl")
landscapes["INTEQNELS"] = gen_inteqnels

include("problems/liarwhd.jl")
landscapes["LIARWHD"] = gen_liarwhd

include("problems/lanczosls.jl")
landscapes["LANCZOS1LS"] = gen_lanczos1ls
landscapes["LANCZOS2LS"] = gen_lanczos2ls
landscapes["LANCZOS3LS"] = gen_lanczos3ls
include("problems/luksan11ls.jl")
landscapes["LUKSAN11LS"] = gen_luksan11ls

include("problems/luksan13ls.jl")
landscapes["LUKSAN13LS"] = gen_luksan13ls

include("problems/luksan15ls.jl")
landscapes["LUKSAN15LS"] = gen_luksan15ls

include("problems/luksan16ls.jl")
landscapes["LUKSAN16LS"] = gen_luksan16ls

include("problems/luksan17ls.jl")
landscapes["LUKSAN17LS"] = gen_luksan17ls

include("problems/luksan21ls.jl")
landscapes["LUKSAN21LS"] = gen_luksan21ls

include("problems/misra1als.jl")
landscapes["MISRA1ALS"] = gen_misra1als

include("problems/misra1bls.jl")
landscapes["MISRA1BLS"] = gen_misra1bls

include("problems/morebv.jl")
landscapes["MOREBV"] = gen_morebv

include("problems/msqrtls.jl")
landscapes["MSQRTALS"] = gen_msqrtals
landscapes["MSQRTBLS"] = gen_msqrtbls

include("problems/ncb20.jl")
landscapes["NCB20"] = gen_ncb20

include("problems/ncb20b.jl")
landscapes["NCB20B"] = gen_ncb20b

include("problems/noncvxun.jl")
include("problems/noncvxu2.jl") # noncvxu2 needs noncvxun to be included
landscapes["NONCVXU2"] = gen_noncvxu2
landscapes["NONCVXUN"] = gen_noncvxun

include("problems/nondia.jl")
landscapes["NONDIA"] = gen_nondia

include("problems/nondquar.jl")
landscapes["NONDQUAR"] = gen_nondquar

include("problems/oscipath.jl")
landscapes["OSCIPATH"] = gen_oscipath

include("problems/palmer.jl")
landscapes["PALMER1C"] = gen_palmer1c
landscapes["PALMER1D"] = gen_palmer1d
landscapes["PALMER2C"] = gen_palmer2c
landscapes["PALMER3C"] = gen_palmer3c
landscapes["PALMER4C"] = gen_palmer4c
landscapes["PALMER5C"] = gen_palmer5c
landscapes["PALMER5D"] = gen_palmer5d
landscapes["PALMER6C"] = gen_palmer6c
landscapes["PALMER7C"] = gen_palmer7c
landscapes["PALMER8C"] = gen_palmer8c

include("problems/penalty1.jl")
landscapes["PENALTY1"] = gen_penalty1

include("problems/penalty2.jl")
landscapes["PENALTY2"] = gen_penalty2

include("problems/penalty3.jl")
landscapes["PENALTY3"] = gen_penalty3

include("problems/powellsg.jl")
landscapes["POWELLSG"] = gen_powellsg

include("problems/power.jl")
landscapes["POWER"] = gen_power

include("problems/qing.jl")
landscapes["QING"] = gen_qing

include("problems/quartc.jl")
landscapes["QUARTC"] = gen_quartc

include("problems/rosenbr.jl")
landscapes["ROSENBR"] = gen_rosenbrock

include("problems/schmvett.jl")
landscapes["SCHMVETT"] = gen_schmvett

include("problems/sinquad2.jl")
landscapes["SINQUAD2"] = gen_sinquad2

include("problems/sparsine.jl")
landscapes["SPARSINE"] = gen_sparsine

include("problems/sparsqur.jl")
landscapes["SPARSQUR"] = gen_sparsqur

include("problems/srosenbr.jl")
landscapes["SROSENBR"] = gen_srosenbr

include("problems/strtchdv.jl")
landscapes["STRTCHDV"] = gen_strtchdv

include("problems/thurberls.jl") # Note: needs hahn1ls.jl included
landscapes["THURBERLS"] = gen_thurberls

include("problems/tointgss.jl")
landscapes["TOINTGSS"] = gen_tointgss

include("problems/tointqor.jl")
landscapes["TOINTQOR"] = gen_tointqor

include("problems/tquartic.jl")
landscapes["TQUARTIC"] = gen_tquartic

include("problems/tridia.jl")
landscapes["TRIDIA"] = gen_tridia

include("problems/trigon1.jl")
landscapes["TRIGON1"] = gen_trigon1

include("problems/vardim.jl")
landscapes["VARDIM"] = gen_vardim

include("problems/vibrbeam.jl")
landscapes["VIBRBEAM"] = gen_vibrbeam

include("problems/woods.jl")
landscapes["WOODS"] = gen_woods