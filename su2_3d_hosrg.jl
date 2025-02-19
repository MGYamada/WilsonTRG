using LinearAlgebra
using SUNRepresentations
using HCubature
using TensorOperations
using SparseArrayKit
using Zygote
using SpecialFunctions

# SU(2) only
const β = 1.0
const box_max = 1
const D_max = 5
const M = 30
const N_sweep = 20

function polar2g(ζ, φ, χ)
    α = sin(ζ) * cis(φ)
    β = cos(ζ) * cis(χ)
    [α -conj(β); β conj(α)]
end

function gauge(T, D_cut, s)
    if s == 'l'
        @tensoropt M_l[a, A, e, E] := T[o, b, c, x, a, d] * T[p, b, c, x, e, d] * T[y, B, C, o, A, D] * T[y, B, C, p, E, D]
        D = size(M_l, 1)
        M_l = reshape(M_l, (D ^ 2, D ^ 2))
        M_l = (M_l + M_l') / 2

        vl, Ul = eigen(M_l; sortby = x -> -x)
        D_new = min(D ^ 2, D_cut)
        inds_new = collect(1:D_new)
        TrunErrLeft = 1.0 - sum(vl[inds_new]) / sum(vl)
        Ul = Ul[:, inds_new]
        Ul = reshape(Ul, (D, D, D_new))
        Ul, TrunErrLeft
    elseif s == 'r'
        @tensoropt M_r[b, B, e, E] := T[o, b, c, x, a, d] * T[p, e, c, x, a, d] * T[y, B, C, o, A, D] * T[y, E, C, p, A, D]
        D = size(M_r, 1)
        M_r = reshape(M_r, (D ^ 2, D ^ 2))
        M_r = (M_r + M_r') / 2

        vr, Ur = eigen(M_r; sortby = x -> -x)
        D_new = min(D ^ 2, D_cut)
        inds_new = collect(1:D_new)
        TrunErrRight = 1.0 - sum(vr[inds_new]) / sum(vr)
        Ur = Ur[:, inds_new]
        Ur = reshape(Ur, (D, D, D_new))
        Ur, TrunErrRight
    elseif s == 'd'
        @tensor M_d[c, C, e, E] := (T[o, b, c, x, a, d] * T[p, b, e, x, a, d]) * (T[y, B, C, o, A, D] * T[y, B, E, p, A, D])
        D = size(M_d, 1)
        M_d = reshape(M_d, (D ^ 2, D ^ 2))
        M_d = (M_d + M_d') / 2

        vd, Ud = eigen(M_d; sortby = x -> -x)
        D_new = min(D ^ 2, D_cut)
        inds_new = collect(1:D_new)
        TrunErrDown = 1.0 - sum(vd[inds_new]) / sum(vd)
        Ud = Ud[:, inds_new]
        Ud = reshape(Ud, (D, D, D_new))
        Ud, TrunErrDown
    elseif s == 'u'
        @tensor M_u[d, D, e, E] := (T[o, b, c, x, a, d] * T[p, b, c, x, a, e]) * (T[y, B, C, o, A, D] * T[y, B, C, p, A, E])
        D = size(M_u, 1)
        M_u = reshape(M_u, (D ^ 2, D ^ 2))
        M_u = (M_u + M_u') / 2

        vu, Uu = eigen(M_u; sortby = x -> -x)
        D_new = min(D ^ 2, D_cut)
        inds_new = collect(1:D_new)
        TrunErrUp = 1.0 - sum(vu[inds_new]) / sum(vu)
        Uu = Uu[:, inds_new]
        Uu = reshape(Uu, (D, D, D_new))
        Uu, TrunErrUp
    end
end

function hotrg(T, D, ::Val{M}) where M
    lnZ = 0.0
    Us = Matrix{Float64}[]
    for k in 1:M
        Ul, TrunErrLeft = gauge(T, D, 'l')
        Ur, TrunErrRight = gauge(T, D, 'r')
        Ud, TrunErrDown = gauge(T, D, 'd')
        Uu, TrunErrUp = gauge(T, D, 'u')
        Ulr = TrunErrLeft < TrunErrRight ? Ul : Ur
        Udu = TrunErrDown < TrunErrUp ? Ud : Uu
        push!(Us, reshape(Ulr, :, size(Ulr, 3)))
        push!(Us, reshape(Udu, :, size(Udu, 3)))
        @tensoropt T[z, j, y, w, i, x] := T[o, b, c, x, a, d] * Ulr[a, A, z] * T[y, B, C, o, A, D] * Ulr[b, B, w] * Udu[c, C, i] * Udu[d, D, j] # fix later
        f = norm(T)
        lnZ += log(f) / (2 ^ k)
        T /= f
    end
    sum = 0.0
    for x in 1:size(T, 1), y in 1:size(T, 2), z in 1:size(T,3)
        sum += T[x, y, z, x, y, z]
    end
    lnZ += log(sum) / (2 ^ M)
    lnZ, ntuple(i -> Us[i], Val(2M))
end

function hosrg(T, D, Us::NTuple{M2, Matrix{Float64}}) where M2
    lnZ = 0.0
    for k in 1:M
        Ulr = reshape(Us[2k - 1], D, D, :)
        Udu = reshape(Us[2k], D, D, :)
        @tensoropt T[z, j, y, w, i, x] := T[o, b, c, x, a, d] * Ulr[a, A, z] * T[y, B, C, o, A, D] * Ulr[b, B, w] * Udu[c, C, i] * Udu[d, D, j] # fix later
        f = norm(T)
        lnZ += log(f) / (2 ^ k)
        T /= f
    end
    sum = 0.0
    for x in 1:size(T, 1), y in 1:size(T, 2), z in 1:size(T,3)
        sum += T[x, y, z, x, y, z]
    end
    lnZ += log(sum) / (2 ^ M)
end

function armillary_CGC(r12, r23, r31, ρ1, m1, ρ2, m2, ρ3, m3)
    A1 = CGC(r31[1], r12[3], ρ1[1])[:, :, :, m1[1]]
    B1 = CGC(r31[2], r12[1], ρ1[1])[:, :, :, m1[1]]
    A2 = CGC(r12[1], r23[3], ρ2[1])[:, :, :, m2[1]]
    B2 = CGC(r12[2], r23[1], ρ2[1])[:, :, :, m2[1]]
    A3 = CGC(r23[1], r31[3], ρ3[1])[:, :, :, m3[1]]
    B3 = CGC(r23[2], r31[1], ρ3[1])[:, :, :, m3[1]]
    A4 = CGC(r31[3], r12[4], ρ1[2])[:, :, :, m1[2]]
    B4 = CGC(r31[4], r12[2], ρ1[2])[:, :, :, m1[2]]
    A5 = CGC(r12[3], r23[4], ρ2[2])[:, :, :, m2[2]]
    B5 = CGC(r12[4], r23[2], ρ2[2])[:, :, :, m2[2]]
    A6 = CGC(r23[3], r31[4], ρ3[2])[:, :, :, m3[2]]
    B6 = CGC(r23[4], r31[2], ρ3[2])[:, :, :, m3[2]]
    @tensoropt A[] := A1[r311, r123, ρ11] * A2[r121, r233, ρ21] * A3[r231, r313, ρ31] * A4[r313, r124, ρ12] * A5[r123, r234, ρ22] * A6[r233, r314, ρ32] * B1[r312, r121, ρ11] * B2[r122, r231, ρ21] * B3[r232, r311, ρ31] * B4[r314, r122, ρ12] * B5[r124, r232, ρ22] * B6[r234, r312, ρ32]
    A[]
end

function main()
    k_max = box_max + 1
    fk = [2k * exp(-β) * besseli(k, β) / β for k in 1:k_max]
    reps = [SUNIrrep(i, 0) for i in 0:box_max]
    indices = NTuple{5, SUNIrrep{2}}[]
    for r1 in reps, r2 in reps, r3 in reps, r4 in reps
        d1 = directproduct(r1, r2)
        d2 = directproduct(r3, r4)
        for k in keys(d1)
            if haskey(d2, k)
                push!(indices, (r1, r2, r3, r4, k)) # outer multiplicity
            end
        end
    end
    L = length(indices)
    T = SparseArray{Float64}(undef, L, L, L, L, L, L)
    for r121 in reps, r231 in reps, r311 in reps, r122 in reps, r232 in reps, r312 in reps, r123 in reps, r233 in reps, r313 in reps, r124 in reps, r234 in reps, r314 in reps
        a1 = directproduct(r311, r123)
        b1 = directproduct(r312, r121)
        a2 = directproduct(r121, r233)
        b2 = directproduct(r122, r231)
        a3 = directproduct(r231, r313)
        b3 = directproduct(r232, r311)
        a4 = directproduct(r313, r124)
        b4 = directproduct(r314, r122)
        a5 = directproduct(r123, r234)
        b5 = directproduct(r124, r232)
        a6 = directproduct(r233, r314)
        b6 = directproduct(r234, r312)
        for r1 in keys(a1)
            if haskey(b1, r1)
                for r2 in keys(a2)
                    if haskey(b2, r2)
                        for r3 in keys(a3)
                            if haskey(b3, r3)
                                for r4 in keys(a4)
                                    if haskey(b4, r4)
                                        for r5 in keys(a5)
                                            if haskey(b5, r5)
                                                for r6 in keys(a6)
                                                    if haskey(b6, r6)
                                                        ind1 = findfirst(isequal((r311, r123, r312, r121, r1)), indices)
                                                        ind2 = findfirst(isequal((r121, r233, r122, r231, r2)), indices)
                                                        ind3 = findfirst(isequal((r231, r313, r232, r311, r3)), indices)
                                                        ind4 = findfirst(isequal((r313, r124, r314, r122, r4)), indices)
                                                        ind5 = findfirst(isequal((r123, r234, r124, r232, r5)), indices)
                                                        ind6 = findfirst(isequal((r233, r314, r234, r312, r6)), indices)
                                                        T[ind1, ind2, ind3, ind4, ind5, ind6] = armillary_CGC([r121, r122, r123, r124], [r231, r232, r233, r234], [r311, r312, r313, r314], [r1, r4], [1, 1], [r2, r5], [1, 1], [r3, r6], [1, 1]) *
                                                        prod(r -> fk[weight(r)[1] + 1], [r121, r122, r123, r124, r231, r232, r233, r234, r311, r312, r313, r314]) ^ (1 / 4) / sqrt(prod(dim, [r1, r2, r3, r4, r5, r6]))
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    Ux = zeros(L, D_max)
    Ux[1:D_max, :] .= Matrix{Float64}(I, D_max, D_max)
    Uy = zeros(L, D_max)
    Uy[1:D_max, :] .= Matrix{Float64}(I, D_max, D_max)
    Uz = zeros(L, D_max)
    Uz[1:D_max, :] .= Matrix{Float64}(I, D_max, D_max)
    Us = Matrix{Float64}[]
    for i in 1:2M
        U = zeros(D_max ^ 2, D_max)
        U[1:D_max, :] .= Matrix{Float64}(I, D_max, D_max)
        push!(Us, U)
    end
    for i in 1:N_sweep
        lnZ, (dUx, dUy, dUz, dUs...) = withgradient(Ux, Uy, Uz, Us...) do Ux, Uy, Uz, Us...
            temp1 = Ux' * reshape(T, L, L ^ 5)
            temp2 = Uy' * reshape(transpose(temp1), L, L ^ 4 * D_max)
            temp3 = Uz' * reshape(transpose(temp2), L, L ^ 3 * D_max ^ 2)
            temp4 = Ux' * reshape(transpose(temp3), L, L ^ 2 * D_max ^ 3)
            temp5 = Uy' * reshape(transpose(temp4), L, L * D_max ^ 4)
            temp6 = Uz' * reshape(transpose(temp5), L, D_max ^ 5)
            hosrg(reshape(Array(temp6), D_max, D_max, D_max, D_max, D_max, D_max), D_max, Us)
        end
        println("HOSRG(", i, "): ", lnZ)
        U, S, V = svd(dUx)
        Ux .= U * V'
        U, S, V = svd(dUy)
        Uy .= U * V'
        U, S, V = svd(dUz)
        Uz .= U * V'
        for i in 1:2M
            U, S, V = svd(dUs[i])
            Us[i] .= U * V'
        end
    end
end

main()