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

function hosrg(T, D, Us::Tuple)
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
    Ud = randn(D, D_max)
    U, S, V = svd(Ud)
    Ud .= U * V'
    Us = ntuple(Val(M)) do k
        Uk = randn(D_max ^ 2, D_max)
        U, S, V = svd(Uk)
        U * V'
    end
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