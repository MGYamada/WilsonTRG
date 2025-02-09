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
    Us = ntuple(Val(M)) do k
        Uk = randn(D_max ^ 2, D_max)
        U, S, V = svd(Uk)
        U * V'
    end
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