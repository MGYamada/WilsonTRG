using LinearAlgebra
using HCubature
using TensorOperations
using SparseArrayKit
using Zygote

# SU(2) only
const β = 1.0
const k_max = 2
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
        @tensor M_l[a, A, e, E] := (T[o, b, c, x, a, d] * T[p, b, c, x, e, d]) * (T[y, B, C, o, A, D] * T[y, B, C, p, E, D])
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
        @tensor M_r[b, B, e, E] := (T[o, b, c, x, a, d] * T[p, e, c, x, a, d]) * (T[y, B, C, o, A, D] * T[y, E, C, p, A, D])
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

function hotrg(T, D, ::Val{M})
    lnZ = 0.0
    Us = Matrix{Float64}[]
    for k in 1:M
        Ul, TrunErrLeft = gauge(T, D, 'l')
        Ur, TrunErrRight = gauge(T, D, 'r')
        Ud, TrunErrDown = gauge(T, D, 'd')
        Uu, TrunErrUp = gauge(T, D, 'u')
        U = [Ul, Ur, Ud, Uu][argmin([TrunErrLeft, TrunErrRight, TrunErrDown, TrunErrUp])]
        push!(Us, reshape(U, :, size(U, 3)))
        @tensoropt T[z, j, y, w, i, x] := T[o, b, c, x, a, d] * U[a, A, z] * T[y, B, C, o, A, D] * U[b, B, w] * U[c, C, i] * U[d, D, j]
        f = norm(T)
        lnZ += log(f) / (2 ^ k)
        T /= f
    end
    sum = 0.0
    for x in 1:size(T, 1), y in 1:size(T, 2), z in 1:size(T,3)
        sum += T[x, y, z, x, y, z]
    end
    lnZ += log(sum) / (2 ^ M)
    lnZ, ntuple(i -> Us[i], Val(M))
end

function hosrg(T, D, Us::NTuple{M, Matrix{Float64}})
    lnZ = 0.0
    for k in 1:M
        U = reshape(Us[k], D, D, :)
        @tensoropt T[z, j, y, w, i, x] := T[o, b, c, x, a, d] * U[a, A, z] * T[y, B, C, o, A, D] * U[b, B, w] * U[c, C, i] * U[d, D, j]
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

function main()
    λk = Float64[]
    for k in 1:k_max
        res, = hcubature(zeros(3), [π / 2, 2π, 2π]) do x
            U = polar2g(x...)
            (1 / (2 * (π ^ 2))) * sin(x[1]) * cos(x[1]) * exp(-(β / 2) * real(tr(I - U))) * sin(k * x[2]) / sin(x[2])
        end
        push!(λk, (1 / k) * res)
    end
    D = sum(k ^ 2 for k in 1:k_max)
    A = SparseArray{Float64}(undef, D, D, D, D)
    k_begin = 0
    for k in 1:k_max
        Ak = SparseArray{Float64}(undef, k, k, k, k, k, k, k, k)
        for a in 1:k, b in 1:k, c in 1:k, d in 1:k
            Ak[a, d, b, a, c, b, d, c] = 1.0
        end
        k² = k ^ 2
        range = (k_begin + 1):(k_begin + k²)
        A[range, range, range, range] .= (λk[k] / k²) .* reshape(Ak, k², k², k², k²)
        k_begin += k²
    end
    B = SparseArray{Float64}(undef, D, D, D, D)
    for i in 1:D
        B[i, i, i, i] = 1.0
    end
    @tensoropt temp[i, j, k, l, m, n, o, p, q, r, s, t] := A[c, d, o, m] * A[e, f, p, k] * A[a, b, n, l] * B[d, c, s, q] * B[e, a, r, i] * B[f, b, t, j]
    D² = D ^ 2
    T = reshape(temp, D², D², D², D², D², D²)
    @tensor ρ[a, b] := T[a, i, j, k, l, m] * T[b, i, j, k, l, m]
    ρ = (ρ + ρ') / 2
    _, vec = eigen(ρ; sortby = x -> -x)
    Ux = vec[:, 1:D_max]
    @tensor ρ[a, b] := T[i, a, j, k, l, m] * T[i, b, j, k, l, m]
    ρ = (ρ + ρ') / 2
    _, vec = eigen(ρ; sortby = x -> -x)
    Uy = vec[:, 1:D_max]
    @tensor ρ[a, b] := T[i, j, a, k, l, m] * T[i, j, b, k, l, m]
    ρ = (ρ + ρ') / 2
    _, vec = eigen(ρ; sortby = x -> -x)
    Uz = vec[:, 1:D_max]
    temp1 = Ux' * reshape(T, D², D² ^ 5)
    temp2 = Uy' * reshape(transpose(temp1), D², D² ^ 4 * D_max)
    temp3 = Uz' * reshape(transpose(temp2), D², D² ^ 3 * D_max ^ 2)
    temp4 = Ux' * reshape(transpose(temp3), D², D² ^ 2 * D_max ^ 3)
    temp5 = Uy' * reshape(transpose(temp4), D², D² * D_max ^ 4)
    temp6 = Uz' * reshape(transpose(temp5), D², D_max ^ 5)
    lnZ, Us = hotrg(reshape(Array(temp6), D_max, D_max, D_max, D_max, D_max, D_max), D_max, Val(M))
    println("HOTRG: ", lnZ)
    for i in 1:N_sweep
        lnZ, (dUx, dUy, dUz, dUs...) = withgradient(Ux, Uy, Uz, Us...) do Ux, Uy, Uz, Us...
            temp1 = Ux' * reshape(T, D², D² ^ 5)
            temp2 = Uy' * reshape(transpose(temp1), D², D² ^ 4 * D_max)
            temp3 = Uz' * reshape(transpose(temp2), D², D² ^ 3 * D_max ^ 2)
            temp4 = Ux' * reshape(transpose(temp3), D², D² ^ 2 * D_max ^ 3)
            temp5 = Uy' * reshape(transpose(temp4), D², D² * D_max ^ 4)
            temp6 = Uz' * reshape(transpose(temp5), D², D_max ^ 5)
            hosrg(reshape(Array(temp6), D_max, D_max, D_max, D_max, D_max, D_max), D_max, Us)
        end
        println("HOSRG(", i, "): ", lnZ)
        U, S, V = svd(dUx)
        Ux .= U * V'
        U, S, V = svd(dUy)
        Uy .= U * V'
        U, S, V = svd(dUz)
        Uz .= U * V'
        for i in 1:M
            U, S, V = svd(dUs[i])
            Us[i] .= U * V'
        end
    end
end

main()