# Communication.jl
#
# 本模块扩展了幽灵区域数据交换功能，支持批量通信（多变量一次性打包传输），
# 异步非阻塞通信以及错误检测和超时机制，适用于高阶差分和多尺度耦合问题的多层幽灵区更新，
# 同时提供通信后处理接口用于性能统计和调试。
#
# 依赖：Distributed, LinearAlgebra, Dates

module Communication

using Distributed
using LinearAlgebra
using Dates  # 用于时间测量
using Base.Threads: Atomic, atomic_add!

export get_border_slice, set_border_slice!, ghost_exchange_3d_batch!, comm_performance_report, call_with_timeout

# 全局通信统计信息（原子更新：:calls 和 :errors 使用 Atomic 类型）
const comm_stats = Dict{Symbol,Any}(
    :total_time  => 0.0,    # 普通数值
    :total_bytes => 0,      # 普通数值
    :calls       => Atomic{Int64}(0),
    :errors      => Atomic{Int64}(0),
)

# 辅助函数：计算数组所占字节数
function array_size_bytes(arr::AbstractArray)
    return sizeof(eltype(arr)) * length(arr)
end

##########################################################################
# 边界切片操作：单变量版本
##########################################################################
function border_slice(field::Array{Float64,3}, direction::Symbol, ghost::NTuple{3,Int})
    gx, gy, gz = ghost
    local_nx = size(field, 1) - 2*gx
    local_ny = size(field, 2) - 2*gy
    local_nz = size(field, 3) - 2*gz
    if direction == :left
         return field[gx+1:2*gx, gy+1:gy+local_ny, gz+1:gz+local_nz]
    elseif direction == :right
         return field[end-2*gx+1:end-gx, gy+1:gy+local_ny, gz+1:gz+local_nz]
    elseif direction == :front
         return field[gx+1:gx+local_nx, gy+1:2*gy, gz+1:gz+local_nz]
    elseif direction == :back
         return field[gx+1:gx+local_nx, end-2*gy+1:end-gy, gz+1:gz+local_nz]
    elseif direction == :bottom
         return field[gx+1:gx+local_nx, gy+1:gy+local_ny, gz+1:2*gz]
    elseif direction == :top
         return field[gx+1:gx+local_nx, gy+1:gy+local_ny, end-2*gz+1:end-gz]
    else
         error("Unknown direction: $direction")
    end
end

function set_border_slice!(field::Array{Float64,3}, direction::Symbol, ghost::NTuple{3,Int}, data::Array{Float64,3})
    gx, gy, gz = ghost
    local_nx = size(field, 1) - 2*gx
    local_ny = size(field, 2) - 2*gy
    local_nz = size(field, 3) - 2*gz
    if direction == :left
         field[1:gx, gy+1:gy+local_ny, gz+1:gz+local_nz] .= data
    elseif direction == :right
         field[end-gx+1:end, gy+1:gy+local_ny, gz+1:gz+local_nz] .= data
    elseif direction == :front
         field[gx+1:gx+local_nx, 1:gy, gz+1:gz+local_nz] .= data
    elseif direction == :back
         field[gx+1:gx+local_nx, end-gy+1:end, gz+1:gz+local_nz] .= data
    elseif direction == :bottom
         field[gx+1:gx+local_nx, gy+1:gy+local_ny, 1:gz] .= data
    elseif direction == :top
         field[gx+1:gx+local_nx, gy+1:gy+local_ny, end-gz+1:end] .= data
    else
         error("Unknown direction: $direction")
    end
    return field
end

##########################################################################
# 批量边界切片操作：多变量版本
##########################################################################
function get_border_slices_batch(direction::Symbol, ghost::NTuple{3,Int}, field_dict::Dict{Symbol,Array{Float64,3}})
    result = Dict{Symbol,Array{Float64,3}}()
    for (fname, field) in field_dict
         result[fname] = begin
             gx, gy, gz = ghost
             local_nx = size(field, 1) - 2*gx
             local_ny = size(field, 2) - 2*gy
             local_nz = size(field, 3) - 2*gz
             if direction == :left
                 field[gx+1:2*gx, gy+1:gy+local_ny, gz+1:gz+local_nz]
             elseif direction == :right
                 field[end-2*gx+1:end-gx, gy+1:gy+local_ny, gz+1:gz+local_nz]
             elseif direction == :front
                 field[gx+1:gx+local_nx, gy+1:2*gy, gz+1:gz+local_nz]
             elseif direction == :back
                 field[gx+1:gx+local_nx, end-2*gy+1:end-gy, gz+1:gz+local_nz]
             elseif direction == :bottom
                 field[gx+1:gx+local_nx, gy+1:gy+local_ny, gz+1:2*gz]
             elseif direction == :top
                 field[gx+1:gx+local_nx, gy+1:gy+local_ny, end-2*gz+1:end-gz]
             else
                 error("Unknown direction: $direction")
             end
         end
    end
    return result
end

function set_border_slices_batch!(field_dict::Dict{Symbol,Array{Float64,3}}, direction::Symbol, ghost::NTuple{3,Int}, data_dict::Dict{Symbol,Array{Float64,3}})
    for (fname, data) in data_dict
         if haskey(field_dict, fname)
             set_border_slice!(field_dict[fname], direction, ghost, data)
         end
    end
    return field_dict
end

##########################################################################
# 超时包装函数：运行函数 f 并在 timeout 秒内等待完成
##########################################################################
function call_with_timeout(f::Function, timeout::Float64)
    fut = @async f()
    t0 = time()
    while !istaskdone(fut)
        if (time() - t0) > timeout
            # 取消任务的调用已移除，因为 Task 对象没有 cancel 方法
            throw(InterruptException("Operation timed out after $timeout seconds"))
        end
        sleep(0.01)
    end
    return fetch(fut)
end

##########################################################################
# 批量幽灵区域数据交换：非阻塞异步通信、错误检测、超时、性能统计
##########################################################################
const opposites = Dict(:left => :right, :right => :left,
                         :front => :back, :back => :front,
                         :bottom => :top, :top => :bottom)

"""
    ghost_exchange_3d_batch!(field_dict::Dict{Symbol,Array{Float64,3}}, domain; timeout=5.0)

对局部子域内多变量的 3D 数据（含 ghost 层）进行批量通信，
每个方向一次性传输所有变量的边界数据。
采用非阻塞异步通信，并引入错误检测和超时机制。
参数：
- field_dict: 字典，键为变量名称 (Symbol)，值为对应的 3D 数组。
- domain: 包含 overlap::NTuple{3,Int} 和 neighbors::Dict{Symbol,Int} 的子域结构体，
          并假定其全局变量 Main.local_fields 指向 field_dict。
- timeout: 超时时间，单位秒，默认 5.0。
返回更新后的 field_dict，同时更新全局通信统计信息。
"""
function ghost_exchange_3d_batch!(field_dict::Dict{Symbol,Array{Float64,3}}, domain; timeout::Float64=5.0)
    # 检查 Main 中是否定义了 local_fields
    if !isdefined(Main, :local_fields)
        Main.eval(Main, :(local_fields = $field_dict))
    else
        Main.local_fields = field_dict
    end

    start_time = time()
    total_bytes = 0
    futures = []

    for (dir, neighbor_pid) in domain.neighbors
        opp_dir = opposites[dir]
        fut = @async begin
            try
                # 显式使用 Communication.get_border_slices_batch
                call_with_timeout(() -> remotecall_fetch(Communication.get_border_slices_batch, neighbor_pid, opp_dir, domain.overlap, Main.local_fields), timeout)
            catch e
                @warn "Error in remote batch call for direction $dir on proc $neighbor_pid: $e"
                atomic_add!(comm_stats[:errors], 1)
                return Dict{Symbol,Array{Float64,3}}()
            end
        end
        push!(futures, (dir, fut))
    end

    for (dir, fut) in futures
        try
            data_dict = fetch(fut)
            for arr in values(data_dict)
                total_bytes += array_size_bytes(arr)
            end
            set_border_slices_batch!(Main.local_fields, dir, domain.overlap, data_dict)
        catch e
            @warn "Error fetching remote data for direction $dir: $e"
            atomic_add!(comm_stats[:errors], 1)
        end
    end

    elapsed = time() - start_time
    comm_stats[:total_time] += elapsed
    comm_stats[:total_bytes] += total_bytes
    atomic_add!(comm_stats[:calls], 1)
    return field_dict
end

##########################################################################
# 通信性能报告接口
##########################################################################
"""
    comm_performance_report() -> Dict{Symbol,Any}

返回当前通信统计信息，包括总通信时间、传输字节数、调用次数、错误次数等。
"""
function comm_performance_report()
    return deepcopy(comm_stats)
end

end  # module Communication
