# DomainDecomposition.jl
#
# 本模块对全局网格进行三维域分解，扩展功能包括：
#  - 多维与非均匀域分解：根据用户提供的权重实现非均匀划分，达到负载均衡。
#  - 重叠区域支持：允许指定各方向的重叠区宽度。
#  - 多级域分解：先进行粗分解，再对每个大块进行细分。
#  - 动态重构接口：可在模拟过程中根据负载变化重新划分子域，实现动态负载均衡。
#  - 与 GridModule 紧密耦合：通过接口生成局部网格供后续求解器使用。
#
# 新增负载均衡策略及通信接口：
#  - 每个 worker 需维护全局变量 local_load（标量）表示局部计算负载，
#    模块通过 remotecall_fetch 采集各域负载，并根据子域在各方向的平均负载计算新权重，
#    然后调用重构接口实现重新划分。
#
# 依赖：Distributed, LinearAlgebra, GridModule

module DomainDecomposition

using Distributed
using LinearAlgebra
using Main.GridModule  # 修改为绝对引用，从 Main 中加载 GridModule

export Domain3D, decompose_grid_3d, reconstruct_decomposition, multi_level_decompose,
       get_local_grid, update_decomposition_with_load

##########################################################################
# 扩展 Domain3D 结构体
##########################################################################

"""
    Domain3D

存储 3D 子域信息，用于并行计算中的域分解。

字段说明：
- global_grid::GridModule.Grid: 全局网格信息。
- proc_id::Int: 分配给该子域的 worker 进程 ID。
- ix, iy, iz::UnitRange{Int}: 子域在全局网格中 x, y, z 方向的索引范围（含重叠区）。
- overlap::NTuple{3,Int}: 每个方向上的重叠区宽度 (ox, oy, oz)。
- neighbors::Dict{Symbol,Int}: 邻居进程信息，键包括 :left, :right, :front, :back, :bottom, :top。
- proc_coord::NTuple{3,Int}: 当前子域在进程网格中的坐标（如 (i,j,k)）。
- level::Int: 分解层级（例如 1 表示粗分解，2 表示细分解）。
"""
struct Domain3D
    global_grid::GridModule.Grid
    proc_id::Int
    ix::UnitRange{Int}
    iy::UnitRange{Int}
    iz::UnitRange{Int}
    overlap::NTuple{3,Int}
    neighbors::Dict{Symbol,Int}
    proc_coord::NTuple{3,Int}
    level::Int
end

##########################################################################
# 非均匀划分工具：动态分区函数
##########################################################################

"""
    dynamic_partition(n::Int, parts::Int; weights::Vector{Float64}=ones(parts)) -> Vector{UnitRange{Int}}

按照给定的权重在 [1, n] 上划分成 parts 个子区间，用于非均匀域分解。
若未提供权重，则默认均匀划分。
"""
function dynamic_partition(n::Int, parts::Int; weights::Vector{Float64}=ones(parts))
    total = sum(weights)
    # 计算目标累计索引（取整）
    cum_targets = [round(Int, n * (sum(weights[1:i]) / total)) for i in 1:parts]
    ranges = Vector{UnitRange{Int}}(undef, parts)
    prev = 0
    for i in 1:parts
        current = max(cum_targets[i], prev + 1)
        ranges[i] = (prev + 1):current
        prev = current
    end
    # 调整最后一个区间以确保覆盖 n
    if last(ranges).stop < n
        ranges[end] = first(ranges[end]):n
    end
    return ranges
end

##########################################################################
# 主函数：三维域分解
##########################################################################

"""
    decompose_grid_3d(grid::GridModule.Grid; overlap::NTuple{3,Int}=(0,0,0), proc_dims::Union{Nothing,NTuple{3,Int}}=nothing, weights::Dict{Symbol,Vector{Float64}}=Dict())

对全局网格进行三维域分解，支持非均匀划分（依据 weights）和重叠区域。
参数：
- grid: 全局网格 (GridModule.Grid)。
- overlap: 重叠区域宽度 (ox, oy, oz)，默认无重叠 (0,0,0)。
- proc_dims: 三维进程排列 (Px, Py, Pz)；若未提供，则根据 worker 数自动分配。
- weights: 一个字典，键为坐标轴符号（例如 :x, :y, :z），
  值为 Vector{Float64}，长度应与对应方向的分块数一致；若未提供则默认均匀划分。
返回值：Domain3D 对象数组，level 设为 1。
"""
function decompose_grid_3d(grid::GridModule.Grid; overlap::NTuple{3,Int}=(0,0,0),
                           proc_dims::Union{Nothing,NTuple{3,Int}}=nothing,
                           weights::Dict{Symbol,Vector{Float64}}=Dict())
    workers_list = sort(workers())
    nprocs_total = length(workers_list)
    if nprocs_total == 0
        error("未检测到 worker 进程，请使用 addprocs() 添加进程。")
    end

    # 根据坐标系统确定维度键
    dim_keys = grid.coordinate_system == :cartesian ? (:x, :y, :z) :
               grid.coordinate_system == :cylindrical ? (:r, :θ, :z) :
               grid.coordinate_system == :spherical ? (:r, :θ, :φ) :
               error("Unsupported coordinate system: $(grid.coordinate_system)")

    # 自动确定进程排列（近似立方体）若未指定
    if proc_dims === nothing
        p = nprocs_total
        Px = floor(Int, cbrt(p))
        Py = Px
        Pz = div(p, Px * Py)
        while Px * Py * Pz < p
            if Px <= Py && Px <= Pz
                Px += 1
            elseif Py <= Px && Py <= Pz
                Py += 1
            else
                Pz += 1
            end
        end
        proc_dims = (Px, Py, Pz)
    else
        if prod(proc_dims) != nprocs_total
            error("指定的 proc_dims $(proc_dims) 与 worker 数量 $nprocs_total 不匹配。")
        end
    end
    dims_proc = (proc_dims[1], proc_dims[2], proc_dims[3])

    # 对每个方向进行非均匀分区，若提供权重则使用 dynamic_partition；否则均匀划分
    partitions = Dict{Symbol, Vector{UnitRange{Int}}}()
    dims_proc_array = (dims_proc[1], dims_proc[2], dims_proc[3])
    for (i, d) in enumerate(dim_keys)
        parts_count = dims_proc_array[i]
        weight_vector = haskey(weights, d) ? weights[d] : ones(parts_count)
        partitions[d] = dynamic_partition(grid.dims[d], parts_count; weights = weight_vector)
    end

    domains = Vector{Domain3D}()
    count = 1
    for k in 1:dims_proc[3]
        for j in 1:dims_proc[2]
            for i in 1:dims_proc[1]
                pid = workers_list[count]
                range_x = partitions[dim_keys[1]][i]
                range_y = partitions[dim_keys[2]][j]
                range_z = partitions[dim_keys[3]][k]
                # 考虑重叠区域，扩展区间但不超出全局网格范围
                ox, oy, oz = overlap
                ix_new = max(first(range_x) - ox, 1) : min(last(range_x) + ox, grid.dims[dim_keys[1]])
                iy_new = max(first(range_y) - oy, 1) : min(last(range_y) + oy, grid.dims[dim_keys[2]])
                iz_new = max(first(range_z) - oz, 1) : min(last(range_z) + oz, grid.dims[dim_keys[3]])
                proc_coord = (i, j, k)
                neighbors = Dict{Symbol,Int}()
                if i > 1
                    neighbors[:left] = workers_list[count - 1]
                end
                if i < dims_proc[1]
                    neighbors[:right] = workers_list[count + 1]
                end
                if j > 1
                    neighbors[:front] = workers_list[count - dims_proc[1]]
                end
                if j < dims_proc[2]
                    neighbors[:back] = workers_list[count + dims_proc[1]]
                end
                if k > 1
                    neighbors[:bottom] = workers_list[count - dims_proc[1]*dims_proc[2]]
                end
                if k < dims_proc[3]
                    neighbors[:top] = workers_list[count + dims_proc[1]*dims_proc[2]]
                end

                push!(domains, Domain3D(grid, pid, ix_new, iy_new, iz_new, overlap, neighbors, proc_coord, 1))
                count += 1
            end
        end
    end
    return domains
end

##########################################################################
# 动态重构接口：重新划分子域（动态负载均衡）
##########################################################################

"""
    reconstruct_decomposition(grid::GridModule.Grid; kwargs...) -> Vector{Domain3D}

根据新的负载或物理梯度信息，重新划分子域，实现动态负载均衡。
接受与 decompose_grid_3d 相同的关键字参数（如 overlap、proc_dims、weights）。
"""
function reconstruct_decomposition(grid::GridModule.Grid; kwargs...)
    return decompose_grid_3d(grid; kwargs...)
end

##########################################################################
# 多级域分解
##########################################################################

"""
    multi_level_decompose(grid::GridModule.Grid, coarse_proc_dims::NTuple{3,Int}, fine_proc_dims::NTuple{3,Int};
                          overlap::NTuple{3,Int}=(0,0,0), coarse_weights::Dict{Symbol,Vector{Float64}}=Dict(),
                          fine_weights::Dict{Symbol,Vector{Float64}}=Dict())
    
先对全局网格按照 coarse_proc_dims 进行粗分解，再对每个粗子域内部根据 fine_proc_dims 进行细分。
返回所有细分后的 Domain3D 数组，其中细分域的 level 字段设为 2。
"""
function multi_level_decompose(grid::GridModule.Grid, coarse_proc_dims::NTuple{3,Int}, fine_proc_dims::NTuple{3,Int};
                               overlap::NTuple{3,Int}=(0,0,0), coarse_weights::Dict{Symbol,Vector{Float64}}=Dict(),
                               fine_weights::Dict{Symbol,Vector{Float64}}=Dict())
    coarse_domains = decompose_grid_3d(grid; overlap = overlap, proc_dims = coarse_proc_dims, weights = coarse_weights)
    fine_domains = Vector{Domain3D}()
    for cd in coarse_domains
        # 针对每个粗子域，根据 cd 的索引范围生成局部子网格参数
        local_limits = Dict{Symbol,Tuple{Float64,Float64}}()
        dim_keys = grid.coordinate_system == :cartesian ? (:x, :y, :z) :
                   grid.coordinate_system == :cylindrical ? (:r, :θ, :z) :
                   grid.coordinate_system == :spherical ? (:r, :θ, :φ) :
                   error("Unsupported coordinate system")
        for d in dim_keys
            coords = grid.coordinates[d]
            if d == dim_keys[1]
                local_limits[d] = (coords[first(cd.ix)], coords[last(cd.ix)])
            elseif d == dim_keys[2]
                local_limits[d] = (coords[first(cd.iy)], coords[last(cd.iy)])
            elseif d == dim_keys[3]
                local_limits[d] = (coords[first(cd.iz)], coords[last(cd.iz)])
            end
        end
        local_grid = GridModule.create_grid(coordinate_system = grid.coordinate_system, limits = local_limits,
                                              spacing = grid.spacing, bc = grid.bc)
        local_fine_domains = decompose_grid_3d(local_grid; overlap = overlap, proc_dims = fine_proc_dims, weights = fine_weights)
        for fd in local_fine_domains
            push!(fine_domains, Domain3D(grid, fd.proc_id, fd.ix, fd.iy, fd.iz, fd.overlap, fd.neighbors, fd.proc_coord, 2))
        end
    end
    return fine_domains
end

##########################################################################
# 与 GridModule 耦合：生成局部网格
##########################################################################

"""
    get_local_grid(domain::Domain3D) -> GridModule.Grid

根据域分解结果，自动生成对应子域的局部网格。
利用 global_grid 的坐标与域的索引范围生成局部坐标数组和局部物理场数据初始化接口。

返回局部网格对象 (GridModule.Grid)。
"""
function get_local_grid(domain::Domain3D)
    grid = domain.global_grid
    dim_keys = grid.coordinate_system == :cartesian ? (:x, :y, :z) :
               grid.coordinate_system == :cylindrical ? (:r, :θ, :z) :
               grid.coordinate_system == :spherical ? (:r, :θ, :φ) :
               error("Unsupported coordinate system")
    local_limits = Dict{Symbol,Tuple{Float64,Float64}}()
    for (i, d) in enumerate(dim_keys)
        coords = grid.coordinates[d]
        if i == 1
            local_limits[d] = (coords[first(domain.ix)], coords[last(domain.ix)])
        elseif i == 2
            local_limits[d] = (coords[first(domain.iy)], coords[last(domain.iy)])
        elseif i == 3
            local_limits[d] = (coords[first(domain.iz)], coords[last(domain.iz)])
        end
    end
    local_grid = GridModule.create_grid(coordinate_system = grid.coordinate_system, limits = local_limits,
                                          spacing = grid.spacing, bc = grid.bc)
    return local_grid
end

##########################################################################
# 新增：负载均衡与通信接口
##########################################################################

"""
    gather_load_metrics(domains::Vector{Domain3D}) -> Dict{NTuple{3,Int},Float64}

从各子域所在的 worker 进程采集负载数据。
假定每个进程上均定义了全局变量 `local_load`（标量）。
返回一个字典，键为子域的 proc_coord，值为对应的负载值。
"""
function gather_load_metrics(domains::Vector{Domain3D})
    load_data = Dict{NTuple{3,Int}, Float64}()
    for d in domains
        # 从对应 worker 进程采集负载数据，若未定义则返回 0.0
        load_val = remotecall_fetch(() -> get(Main, :local_load, 0.0), d.proc_id)
        load_data[d.proc_coord] = load_val
    end
    return load_data
end

"""
    compute_weights_from_load(load_data::Dict{NTuple{3,Int},Float64}, proc_dims::NTuple{3,Int}) -> Dict{Symbol,Vector{Float64}}

根据各子域负载数据，按坐标方向计算新的权重向量。
对于每个维度，取相同索引的子域负载平均值作为该分块的权重。
返回一个字典，键为 :x, :y, :z，对应的值为权重向量。
"""
function compute_weights_from_load(load_data::Dict{NTuple{3,Int},Float64}, proc_dims::NTuple{3,Int})
    Px, Py, Pz = proc_dims
    weight_x = zeros(Px)
    count_x = zeros(Int, Px)
    weight_y = zeros(Py)
    count_y = zeros(Int, Py)
    weight_z = zeros(Pz)
    count_z = zeros(Int, Pz)
    
    for (coord, load) in load_data
        i, j, k = coord
        weight_x[i] += load
        count_x[i] += 1
        weight_y[j] += load
        count_y[j] += 1
        weight_z[k] += load
        count_z[k] += 1
    end
    
    for i in 1:Px
        if count_x[i] > 0
            weight_x[i] /= count_x[i]
        else
            weight_x[i] = 1.0
        end
    end
    for j in 1:Py
        if count_y[j] > 0
            weight_y[j] /= count_y[j]
        else
            weight_y[j] = 1.0
        end
    end
    for k in 1:Pz
        if count_z[k] > 0
            weight_z[k] /= count_z[k]
        else
            weight_z[k] = 1.0
        end
    end
    return Dict(:x => weight_x, :y => weight_y, :z => weight_z)
end

"""
    update_decomposition_with_load(grid::GridModule.Grid, current_domains::Vector{Domain3D};
                                     overlap::NTuple{3,Int}=(0,0,0), proc_dims::Union{Nothing,NTuple{3,Int}}=nothing)
根据各子域的负载数据重新划分子域，实现动态负载均衡。
若 proc_dims 未指定，则根据当前子域的最大 proc_coord 计算。
返回新的 Domain3D 数组。
"""
function update_decomposition_with_load(grid::GridModule.Grid, current_domains::Vector{Domain3D};
                                          overlap::NTuple{3,Int}=(0,0,0), proc_dims::Union{Nothing,NTuple{3,Int}}=nothing)
    if proc_dims === nothing
        # 根据当前子域的最大 proc_coord 得到进程排列
        max_i = maximum(d.proc_coord[1] for d in current_domains)
        max_j = maximum(d.proc_coord[2] for d in current_domains)
        max_k = maximum(d.proc_coord[3] for d in current_domains)
        proc_dims = (max_i, max_j, max_k)
    end
    # 采集各子域负载数据
    load_data = gather_load_metrics(current_domains)
    # 根据负载数据计算新的权重
    new_weights = compute_weights_from_load(load_data, proc_dims)
    # 调用重构接口重新划分子域
    new_domains = decompose_grid_3d(grid; overlap = overlap, proc_dims = proc_dims, weights = new_weights)
    return new_domains
end

end  # module DomainDecomposition
