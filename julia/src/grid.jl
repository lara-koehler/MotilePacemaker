using FFTW

"""
    Grid

Precomputed periodic spectral grid: coordinate meshes, squared wavenumbers, and
FFT plans. Built once per simulation (the Python prototype rebuilt the pacemaker
meshgrid on every single timestep — a real performance bug fixed here).
"""
struct Grid{PF,PI}
    L::Float64
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    X::Matrix{Float64}
    Y::Matrix{Float64}
    k2::Matrix{Float64}
    fft_plan::PF
    ifft_plan::PI
end

"""
    fftfreq(n, d) -> Vector{Float64}

Discrete Fourier sample frequencies, matching `numpy.fft.fftfreq(n, d)`:
`[0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)` (even `n`; analogous for odd `n`).
"""
function fftfreq(n::Int, d::Real)
    val = 1.0 / (n * d)
    freqs = Vector{Float64}(undef, n)
    npos = (n - 1) ÷ 2 + 1
    freqs[1:npos] .= (0:(npos-1)) .* val
    freqs[(npos+1):end] .= (-(n ÷ 2):-1) .* val
    return freqs
end

function make_grid(L::Real, nx::Int, ny::Int)
    dx = L / nx
    dy = L / ny
    xs = (0:(nx-1)) .* dx
    ys = (0:(ny-1)) .* dy
    X = [xs[i] for i in 1:nx, j in 1:ny]
    Y = [ys[j] for i in 1:nx, j in 1:ny]

    kx = fftfreq(nx, dx) .* 2π
    ky = fftfreq(ny, dy) .* 2π
    k2 = [kx[i]^2 + ky[j]^2 for i in 1:nx, j in 1:ny]

    dummy = zeros(ComplexF64, nx, ny)
    fft_plan = plan_fft(dummy)
    ifft_plan = plan_ifft(dummy)

    return Grid(Float64(L), nx, ny, dx, dy, X, Y, k2, fft_plan, ifft_plan)
end
