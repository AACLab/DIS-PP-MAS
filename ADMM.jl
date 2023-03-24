"Parallel version"
function ADMM!(cen::central, robo::Vector{robot}, H::Int64, Neb::Vector{Vector{Int64}}, Jo::Vector{Float64}; 
                MAX_ITER = 200, thres = 1e-1)

    N = cen.N

    ξ     = Vector{opt_cl}(undef,N)
    w     = Vector{opt_cl}(undef,N)
    λ     = Matrix{opt_cl}(undef,N,N)
    vx    = Vector{Vector}(undef,N)
    vy    = Vector{Vector}(undef,N)

    # Initial position setup for all vehicles
    InitX = Matrix{Float64}(undef, H, N)
    InitY = Matrix{Float64}(undef, H, N)
    for i in 1:N
        InitX[:,i] = robo[i].z[1]*ones(H)
        InitY[:,i] = robo[i].z[2]*ones(H)
    end


    for i in 1:N
        w[i] = opt_cl(InitX, InitY)
        for j in 1:N
            λ[i,j] = opt_cl(zeros(H,N), zeros(H,N))
        end
    end


    Jn = zeros(N)

    t0 = time_ns()
    for n in 1:MAX_ITER
        #println("Iterative step $l")
        # Parallel manner in each vehicle
        ρ = 1.1^n

        for i in 1:N
            ξ[i] = Local_Update!(robo[i], cen, i, H, w, λ[i,:], Neb, ρ, Jo[i])
        end
 
        for i in 1:N
            w[i], vx[i], vy[i], Jn[i] = Constraint_Update!(robo[i], cen, i, H, w, ξ, λ[:,i], Neb, ρ, Jo[i])
        end

        # Update dual variable
        for i in 1:N
            for j in Neb[i]
                λ[i,j].Nx = λ[i,j].Nx + ρ*(ξ[i].Nx - w[j].Nx)
                λ[i,j].Ny = λ[i,j].Ny + ρ*(ξ[i].Ny - w[j].Ny)
            end
        end

        ter = 0
        
        for i in 1:N
            ter = ter + norm(vec(ξ[i].Nx - w[i].Nx))
            ter = ter + norm(vec(ξ[i].Ny - w[i].Ny))
        end

        ter = round(ter, digits = 4)
        println("Step $n and error $ter")
        
        if ter <= thres
            break
        end
    end

    t1 = time_ns()
    dt = (t1-t0)/1e9
    dt = round(dt/N; digits = 3)
    println("Average time for each: $dt (s)")
    println("")

    return w, vx, vy, Jn
end



function Local_Update!(robo::robot, cen::central, ii::Int64, H::Int64, w::Vector{opt_cl}, λ::Vector{opt_cl}, Neb::Vector{Vector{Int64}}, ρ::Float64, 
    Jo::Float64)
    # Find pairs of robots that need collision avoidance
    N   = cen.N

    # robo.opti = JuMP.Model(() -> CPLEX.Optimizer(GUROBI_ENV))
    robo.opti = Model(CPLEX.Optimizer)
    set_optimizer_attribute(robo.opti, "CPX_PARAM_EPINT", 1e-8)
    set_silent(robo.opti)


    # Variables
    Nx = @variable(robo.opti, [1:H, 1:N]) # slack variable
    Ny = @variable(robo.opti, [1:H, 1:N]) # slack variable

    # Objective function
    J  = 0

    for j in Neb[ii]
        J = J + ρ/2*dot(vec(Nx - w[j].Nx + λ[j].Nx/ρ), vec(Nx - w[j].Nx + λ[j].Nx/ρ)) + ρ/2*dot(vec(Ny - w[j].Ny + λ[j].Ny/ρ), vec(Ny - w[j].Ny + λ[j].Ny/ρ))
    end

    @objective(robo.opti, MOI.MIN_SENSE, J)
    status = JuMP.optimize!(robo.opti)

    return opt_cl(JuMP.value.(Nx),  JuMP.value.(Ny))
end





function Constraint_Update!(robo::robot, cen::central, ii::Int64, H::Int64, w::Vector{opt_cl}, ξ::Vector{opt_cl}, λ::Vector{opt_cl}, 
                            Neb::Vector{Vector{Int64}}, ρ::Float64, Jo::Float64)
    # Find pairs of robots that need collision avoidance
    N   = cen.N
    T   = robo.T
    τ   = robo.τ
    xF  = robo.pF[1]
    yF  = robo.pF[2]

    # robo.opti = JuMP.Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    robo.opti = Model(CPLEX.Optimizer)
    set_optimizer_attribute(robo.opti, "CPX_PARAM_EPINT", 1e-8)
    set_silent(robo.opti)


    # Variables
    ux = @variable(robo.opti, [1:T])
    uy = @variable(robo.opti, [1:T])
    vx = @variable(robo.opti, [1:T])
    vy = @variable(robo.opti, [1:T])
    Nx = @variable(robo.opti, [1:H, 1:N]) # slack variable
    Ny = @variable(robo.opti, [1:H, 1:N]) # slack variable

    b1 = @variable(robo.opti, [1:H, 1:N],   Bin)
    b2 = @variable(robo.opti, [1:H, 1:N],   Bin)


    @constraint(robo.opti, Nx[H,ii] == xF)
    # @constraint(robo.opti, Ny[H,ii] == yF)

    ϵ  = 0.1
    R  = 100
    for j in ii+1:cen.N
        for h in 1:H
            @constraint(robo.opti, Nx[h,ii] - Nx[h,j] >= 2*robo.Ra + ϵ - R*(    b1[h,j] + b2[h,j]))
            @constraint(robo.opti, Nx[h,ii] - Nx[h,j] >= 2*robo.Ra + ϵ - R*(1 - b1[h,j] + b2[h,j]))
            @constraint(robo.opti, Ny[h,ii] - Ny[h,j] >= 2*robo.Ra + ϵ - R*(1 + b1[h,j] - b2[h,j]))
            @constraint(robo.opti, Ny[h,ii] - Ny[h,j] >= 2*robo.Ra + ϵ - R*(2 - b1[h,j] - b2[h,j]))             
        end
    end

    # Constraints
    @constraints(robo.opti, begin
        -robo.u_max .<= ux .<= robo.u_max
        -robo.u_max .<= uy .<= robo.u_max
         robo.v_min .<= vx .<= robo.v_max
         robo.v_min .<= vy .<= robo.v_max
    end)


    for h in 1:T
        @constraint(robo.opti, robo.x_min + robo.Ra .<= Nx[h,ii] .<= robo.x_max - robo.Ra)
        @constraint(robo.opti, robo.y_min + robo.Ra .<= Ny[h,ii] .<= robo.y_max - robo.Ra)
    end

    # Dynamic Constraints
    for h in 1:T
        if h == 1
            @constraint(robo.opti, Nx[h,ii] - robo.z[1]  - τ*robo.z[3] - τ^2*ux[h] == 0)
            @constraint(robo.opti, Ny[h,ii] - robo.z[2]  - τ*robo.z[4] - τ^2*uy[h] == 0)
            @constraint(robo.opti, vx[h] - robo.z[3]  - τ*ux[h]                 == 0)
            @constraint(robo.opti, vy[h] - robo.z[4]  - τ*uy[h]                 == 0)
        else
            @constraint(robo.opti, Nx[h,ii] - Nx[h-1,ii]  - τ*vx[h-1] - τ^2*ux[h] == 0)
            @constraint(robo.opti, Ny[h,ii] - Ny[h-1,ii]  - τ*vy[h-1] - τ^2*uy[h] == 0)
            @constraint(robo.opti, vx[h] - vx[h-1] - τ*ux[h]               == 0)
            @constraint(robo.opti, vy[h] - vy[h-1] - τ*uy[h]               == 0)  
        end
    end
    # @variable(robo.opti, Jn)

    # Objective function
    J  = 0

    for j in Neb[ii]
        J = J + ρ/2*dot(vec(ξ[j].Nx - Nx + λ[j].Nx/ρ), vec(ξ[j].Nx - Nx + λ[j].Nx/ρ)) + ρ/2*dot(vec(ξ[j].Ny - Ny + λ[j].Ny/ρ), vec(ξ[j].Ny - Ny + λ[j].Ny/ρ))
    end
    @objective(robo.opti, MOI.MIN_SENSE, J)
    
    # Now solve and get results
    status = JuMP.optimize!(robo.opti)

    Jn = sum((JuMP.value.(Nx)[h,ii] - xF)^2 + (JuMP.value.(Ny)[h,ii] - yF)^2 for h in 1:H)
    
    return opt_cl(JuMP.value.(Nx),  JuMP.value.(Ny)), JuMP.value.(vx), JuMP.value.(vy), Jn
end
