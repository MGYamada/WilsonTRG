using LinearAlgebra
using HCubature
using OMEinsum

# SU(2) only
const β = 1.0
const k_max = 5 # spin-2

function polar2g(ζ, φ, χ)
    α = sin(ζ) * cis(φ)
    β = cos(ζ) * cis(χ)
    [α -conj(β); β conj(α)]
end

function gauge(T, D_cut, s)
    if s == 'l'
        M_l = ein"(zbxa, wbxc), (yβwγ, yβzα) -> aαcγ"(T, T, T, T)
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
        M_r = ein"(zaxb, wcxb), (yγwβ, yαzβ) -> aαcγ"(T, T, T, T)
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

function hotrg(T, Dcut, Niter)
    lnZ = 0.0
    for k in 1:Niter
        Ul, TrunErrLeft = gauge(T, Dcut, 'l')
        Ur, TrunErrRight = gauge(T, Dcut, 'r')
        U = TrunErrLeft < TrunErrRight ? Ul : Ur
        T = ein"((obxa, aαz), yβoα), bβw -> zywx"(T, U, T, U)
        f = norm(T)
        lnZ += log(f) / (2 ^ k)
        T /= f
    end
    sum = 0.0
    for x in 1:size(T, 1), y in 1:size(T, 2)
        sum += T[x, y, x, y]
    end
    lnZ += log(sum) / (2 ^ Niter)
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
    T = zeros(D, D, D, D)
    k_begin = 0
    for k in 1:k_max
        Tk = zeros(k, k, k, k, k, k, k, k)
        for a in 1:k, b in 1:k, c in 1:k, d in 1:k
            Tk[a, d, b, a, c, b, d, c] = 1.0
        end
        k² = k ^ 2
        range = (k_begin + 1):(k_begin + k²)
        T[range, range, range, range] .= (λk[k] / k²) .* reshape(Tk, k², k², k², k²)
        k_begin += k²
    end
    println(hotrg(T, D, 20))
end

main()