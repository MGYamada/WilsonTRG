using LinearAlgebra
using HCubature
using TensorOperations
using SparseArrayKit
using Zygote

# SU(2) only
const β = 1.0
const k_max = 5
const D_max = 10
const M = 20
const N_sweep = 20

function polar2g(ζ, φ, χ)
    α = sin(ζ) * cis(φ)
    β = cos(ζ) * cis(χ)
    [α -conj(β); β conj(α)]
end

function gauge(T, D_cut, s)
    if s == 'l'
        @tensor M_l[a, A, c, C] := (T[z, b, x, a] * T[w, b, x, c]) * (T[y, B, w, C] * T[y, B, z, A])
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
        @tensor M_r[a, A, c, C] := (T[z, a, x, b] * T[w, c, x, b]) * (T[y, C, w, B] * T[y, A, z, B])
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
    end
end

function hotrg(T, D, ::Val{M})
    lnZ = 0.0
    Us = Matrix{Float64}[]
    for k in 1:M
        Ul, TrunErrLeft = gauge(T, D, 'l')
        Ur, TrunErrRight = gauge(T, D, 'r')
        U = TrunErrLeft < TrunErrRight ? Ul : Ur
        push!(Us, reshape(U, :, size(U, 3)))
        @tensoropt T[z, y, w, x] := T[o, b, x, a] * U[a, A, z] * T[y, B, o, A] * U[b, B, w]
        f = norm(T)
        lnZ += log(f) / (2 ^ k)
        T /= f
    end
    sum = 0.0
    for x in 1:size(T, 1), y in 1:size(T, 2)
        sum += T[x, y, x, y]
    end
    lnZ += log(sum) / (2 ^ M)
    lnZ, ntuple(i -> Us[i], Val(M))
end

function hosrg(T, D, Us::NTuple{M, Matrix{Float64}})
    lnZ = 0.0
    for k in 1:M
        U = reshape(Us[k], D, D, :)
        @tensoropt T[z, y, w, x] := T[o, b, x, a] * U[a, A, z] * T[y, B, o, A] * U[b, B, w]
        f = norm(T)
        lnZ += log(f) / (2 ^ k)
        T /= f
    end
    sum = 0.0
    for x in 1:size(T, 1), y in 1:size(T, 2)
        sum += T[x, y, x, y]
    end
    lnZ += log(sum) / (2 ^ M)
end

function main(::Val{M}) where M
    λk = Float64[]
    for k in 1:k_max
        res, = hcubature(zeros(3), [π / 2, 2π, 2π]) do x
            U = polar2g(x...)
            (1 / (2 * (π ^ 2))) * sin(x[1]) * cos(x[1]) * exp(-(β / 2) * real(tr(I - U))) * sin(k * x[2]) / sin(x[2])
        end
        push!(λk, (1 / k) * res)
    end
    D = sum(k ^ 2 for k in 1:k_max)
    T = SparseArray{Float64}(undef, D, D, D, D)
    k_begin = 0
    for k in 1:k_max
        Tk = SparseArray{Float64}(undef, k, k, k, k, k, k, k, k)
        for a in 1:k, b in 1:k, c in 1:k, d in 1:k
            Tk[a, d, a, b, b, c, d, c] = 1.0
        end
        k² = k ^ 2
        range = (k_begin + 1):(k_begin + k²)
        T[range, range, range, range] .= (λk[k] ^ 2 / k) .* reshape(Tk, k², k², k², k²)
        k_begin += k²
    end
    @tensor ρ[a, b] := T[a, i, j, k] * T[b, i, j, k]
    ρ = (ρ + ρ') / 2

    _, vec = eigen(ρ; sortby = x -> -x)
    Ud = vec[:, 1:D_max]
    temp1 = Ud' * reshape(T, D, D ^ 3)
    temp2 = Ud' * reshape(transpose(temp1), D, D ^ 2 * D_max)
    temp3 = Ud' * reshape(transpose(temp2), D, D * D_max ^ 2)
    temp4 = Ud' * reshape(transpose(temp3), D, D_max ^ 3)
    lnZ, Us = hotrg(reshape(Array(temp4), D_max, D_max, D_max, D_max), D_max, Val(M))
    println("HOTRG: ", lnZ)
    for i in 1:N_sweep
        lnZ, (dUd, dUs...) = withgradient(Ud, Us...) do Ud, Us...
            temp1 = Ud' * reshape(T, D, D ^ 3)
            temp2 = Ud' * reshape(transpose(temp1), D, D ^ 2 * D_max)
            temp3 = Ud' * reshape(transpose(temp2), D, D * D_max ^ 2)
            temp4 = Ud' * reshape(transpose(temp3), D, D_max ^ 3)
            hosrg(reshape(Array(temp4), D_max, D_max, D_max, D_max), D_max, Us)
        end
        println("HOSRG(", i, "): ", lnZ)
        U, S, V = svd(dUd)
        Ud .= U * V'
        for i in 1:M
            U, S, V = svd(dUs[i])
            Us[i] .= U * V'
        end
    end
end

main(Val(M))