using TOML
using HDF5

"""
    build_simulation_config(raw::Dict) -> (cfg::SimulationConfig, sources::Dict)

Build a `SimulationConfig` from an already-parsed TOML dict. `sources` is the
raw `[sources]` table (source count/arrangement — consumed by
[`initialize_positions`](@ref), not part of the physics config itself).

Split out from [`load_config`](@ref) so a parameter scan can parse a base
config once, apply [`apply_override!`](@ref) for each swept key, and build a
`SimulationConfig` from the *modified* dict without going back through a file.
"""
function build_simulation_config(raw::Dict)
    grid = raw["grid"]
    timecfg = raw["time"]
    chem = raw["chemistry"]
    coup = raw["coupling"]
    mechc = raw["mechanics"]
    noisec = get(raw, "noise", Dict{String,Any}())
    sourcesc = raw["sources"]

    kinetics = FHNKinetics(chem["tau_u"], chem["tau_v"], chem["a"], chem["b"], chem["I0"])
    model = ReactionDiffusionModel(kinetics, chem["Du"], chem["Dv"])

    coupling = PacemakerCoupling(
        coup["pacemaker_strength"], coup["pacemaker_width"],
        get(coup, "cutoff_factor", 4.0),
    )

    # by default, cap each pair's force so a single explicit-Euler step can't
    # move a source more than half its own WCA radius (sigma) -- otherwise a
    # stiff enough close encounter can "tunnel" through the repulsive wall in
    # one step; see MechanicalParams' max_force docs.
    default_max_force = 0.5 * mechc["sigma"] / (timecfg["dt"] * mechc["mobility"])
    mech = MechanicalParams(
        mechc["mobility"], mechc["sigma"], mechc["epsilon_LJ"],
        mechc["interaction_strength"], mechc["interaction_range"],
        mechc["u_threshold"], Symbol(mechc["reciprocity"]);
        rcut_far_factor=get(mechc, "rcut_far_factor", 5.0),
        rcut_near_factor=get(mechc, "rcut_near_factor", 0.05),
        verlet_skin=get(mechc, "verlet_skin", 0.5 * mechc["interaction_range"]),
        max_force=get(mechc, "max_force", default_max_force),
    )

    field_sigma_u = get(noisec, "field_sigma_u", 0.0)
    field_sigma_v = get(noisec, "field_sigma_v", 0.0)
    field_noise = (field_sigma_u > 0 || field_sigma_v > 0) ?
                  AdditiveFieldNoise(field_sigma_u, field_sigma_v) : NoNoise()

    position_sigma = get(noisec, "position_sigma", 0.0)
    position_noise = position_sigma > 0 ? PositionalNoise(position_sigma) : nothing

    cfg = SimulationConfig(
        L=grid["L"], nx=grid["nx"], ny=grid["ny"],
        dt=timecfg["dt"], n_steps=timecfg["n_steps"], save_every=timecfg["save_every"],
        source_save_every=get(timecfg, "source_save_every", 1),
        model=model, coupling=coupling, mech=mech,
        field_noise=field_noise, position_noise=position_noise,
        seed=get(timecfg, "seed", 1),
    )

    return cfg, sourcesc
end

"""
    load_config(path) -> (cfg::SimulationConfig, sources::Dict, raw::Dict)

Parse a TOML run configuration into a `SimulationConfig` via
[`build_simulation_config`](@ref). `raw` is the full parsed TOML, kept for
provenance (written verbatim into the HDF5 output as a string attribute).
"""
function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    cfg, sourcesc = build_simulation_config(raw)
    return cfg, sourcesc, raw
end

"""
    apply_override!(raw::Dict, dotted_key::AbstractString, value)

Set `raw[k1][k2]...[kn] = value` where `dotted_key = "k1.k2...kn"` (e.g.
`"mechanics.mobility"`), creating intermediate `Dict`s as needed. Used to
apply one parameter-scan column's value onto a base config's raw TOML dict
before calling [`build_simulation_config`](@ref).
"""
function apply_override!(raw::Dict, dotted_key::AbstractString, value)
    parts = split(dotted_key, '.')
    d = raw
    for p in parts[1:end-1]
        d = get!(() -> Dict{String,Any}(), d, String(p))
    end
    d[String(parts[end])] = value
    return raw
end

"""
    parse_scan_value(s::AbstractString)

Parse a command-line parameter-scan value string into a `Bool`, `Int`, or
`Float64` if it looks like one, else leave it as a `String` (e.g. `"0.5"` ->
`0.5::Float64`, `"17"` -> `17::Int`, `"nonreciprocal"` -> unchanged).
"""
function parse_scan_value(s::AbstractString)
    lower = lowercase(s)
    lower == "true" && return true
    lower == "false" && return false
    int_val = tryparse(Int, s)
    int_val !== nothing && return int_val
    float_val = tryparse(Float64, s)
    float_val !== nothing && return float_val
    return s
end

"""
    save_run(path, result, config_text)

Write a `SimulationResult` to an HDF5 file at `path`, alongside the exact TOML
config text used to produce it (stored as a string attribute, for provenance).
"""
function save_run(path::AbstractString, result::SimulationResult, config_text::AbstractString)
    mkpath(dirname(path))
    h5open(path, "w") do file
        file["times"] = result.times
        file["u_field_stack"] = result.u_field_stack
        file["v_field_stack"] = result.v_field_stack
        file["x_source"] = result.x_source
        file["y_source"] = result.y_source
        file["u_source"] = result.u_source
        file["v_source"] = result.v_source
        attributes(file)["config_toml"] = String(config_text)
    end
    return path
end

"""
    load_run(path) -> NamedTuple

Read back everything written by `save_run`.
"""
function load_run(path::AbstractString)
    h5open(path, "r") do file
        return (
            times=read(file["times"]),
            u_field_stack=read(file["u_field_stack"]),
            v_field_stack=read(file["v_field_stack"]),
            x_source=read(file["x_source"]),
            y_source=read(file["y_source"]),
            u_source=read(file["u_source"]),
            v_source=read(file["v_source"]),
            config_toml=read(attributes(file)["config_toml"]),
        )
    end
end
