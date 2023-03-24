push!(LOAD_PATH, ".")

import Pkg
Pkg.activate(".")
# Pkg.instantiate()


using LinearAlgebra, JuMP, Plots, Gurobi, CPLEX
using CSV, DataFrames, Dates, MAT

# if !(@isdefined(GUROBI_ENV))
#     const GUROBI_ENV = Gurobi.Env()
# end


ENV["CPLEX_STUDIO_BINARIES"] = "C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio221\\cplex\\bin\\x64_win64"


include("Structure.jl")
include("ADMM.jl")
include("Initiate.jl")


# Initiate 
# Central unit: number of vehicles N, number of obstacle n_obs and sampling time ts
# Vehicle: maximum sampling times
# Obstacle 


st        = 0.5 # Sample time
H         = 8   # Horizontal prediction
sim_times = 60

cen, robo, obs, Fm = Initiate2!(st,H)

N = cen.N

traj     = Vector{opt_cl}(undef, N)  


InitX = Matrix{Float64}(undef, H, N)
InitY = Matrix{Float64}(undef, H, N)
for i in 1:N
    InitX[:,i] = robo[i].z[1]*ones(H)
    InitY[:,i] = robo[i].z[2]*ones(H)
    traj[i]    = opt_cl(InitX, InitY)
end

J = zeros(cen.N)
#--------------------------#
#Network define
Neb = Vector{Vector{Int64}}(undef,N)
for i in 1:N
    if i == 1
        Neb[i] = [N,   i, i+1]
    elseif i == N
        Neb[i] = [i-1, i, 1]
    else
        Neb[i] = [i-1, i, i+1]
    end
end
#--------------------------#


traj, vx, vy, J = ADMM!(cen, robo, H, Neb, J)



display(traj[1].Nx - traj[2].Nx)




function circleShape(x,y,r)
    θ = LinRange(0, 2*π, 100)
    return x .+ r*cos.(θ), y .+ r*sin.(θ)
end


color = ["blue", "green", "pink", "yellow"]

for h = 1:H
    p = plot()
    for i in 1:cen.N
        plot!(circleShape(traj[i].Nx[h,i], traj[i].Ny[h,i], robo[i].Ra), 
                            seriestype = [:shape,], legend = false, fillcolor = color[i], fillalpha = 1.0, tickfont = "Arial")
    end
    xlims!(0, 10)
    ylims!(0, 10)
    display(p)
end







