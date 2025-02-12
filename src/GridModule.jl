module GridModule

using LinearAlgebra
using HDF5
using TOML
using YAML

# 引入 IOManager 中的相关功能
using Main.IOManager
using Main.EOSModule

export Grid, create_grid, apply_boundary_conditions,
       init_physical_fields!, update_physical_field!,
       read_config, get_refinement_level, update_refinement_level,
       refine_grid_combined!, refine_grid_by_value!, refine_grid!

##########################################################################
# 网格数据结构定义
##########################################################################

"""
    Grid

存储模拟网格信息的结构体，支持多坐标系统、自适应网格（AMR）、多种边界条件，
以及物理场数据和配置参数。

字段说明：
- coordinate_system::Symbol  
    坐标系统，支持 :cartesian、:cylindrical、:spherical。
- limits::Dict{Symbol,Tuple{Float64,Float64}}  
    各方向的域边界，例如笛卡尔坐标下包含 :x, :y, :z。
- spacing::Dict{Symbol,Float64}  
    各方向的基础网格间距，用于生成均匀网格。
- coordinates::Dict{Symbol,Vector{Float64}}  
    各方向生成的网格坐标数组。
- dims::Dict{Symbol,Int}  
    各方向上的网格点数。
- bc::Dict{Symbol,Any}  
    边界条件设置，每个键对应一侧边界（如 :xlow、:xhigh），可取预定义类型（如 :Dirichlet、:Neumann、:Absorbing、:Periodic、:Mixed）或带参数的元组。
- adaptive_params::Dict{Symbol,Tuple{Float64,Float64,Float64,Float64}}  
    自适应网格参数，格式为 (refine_start, refine_end, fine_spacing, coarse_spacing)，用于指定局部细化区域。
- custom_bc::Dict{Symbol,Function}  
    用户自定义的边界处理函数，键与 bc 中对应，若存在则优先使用。
- physical_fields::Dict{Symbol,Any}  
    存储物理场数据，如密度、压力、磁场、电场等，数据类型由用户自行定义（一般为数组）。
- config::Dict{Symbol,Any}  
    存储从外部文件读取的配置参数。
- refinement_level::Int  
    当前网格的细化级别，用于动态调整物理模型（如EOS）。
"""
struct Grid
    coordinate_system::Symbol
    limits::Dict{Symbol, Tuple{Float64, Float64}}
    spacing::Dict{Symbol, Float64}
    coordinates::Dict{Symbol, Vector{Float64}}
    dims::Dict{Symbol, Int}
    bc::Dict{Symbol, Any}
    adaptive_params::Dict{Symbol, Tuple{Float64, Float64, Float64, Float64}}
    custom_bc::Dict{Symbol, Function}
    physical_fields::Dict{Symbol, Any}
    config::Dict{Symbol, Any}
    refinement_level::Int  # 当前网格细化级别
end

##########################################################################
# 网格生成函数
##########################################################################

function create_grid(; coordinate_system::Symbol = :cartesian,
                     limits::Dict{Symbol, Tuple{Float64, Float64}},
                     spacing::Dict{Symbol, Float64},
                     bc::Dict{Symbol, Any} = Dict{Symbol, Any}(),
                     adaptive_params::Dict{Symbol, Tuple{Float64, Float64, Float64, Float64}} = Dict{Symbol, Tuple{Float64, Float64, Float64, Float64}}(),
                     stretch_funcs::Dict{Symbol, Function} = Dict{Symbol, Function}(),
                     custom_bc::Dict{Symbol, Function} = Dict{Symbol, Function}(),
                     refinement_level::Int = 1)  # 允许指定初始细化级别
    coordinates = Dict{Symbol, Vector{Float64}}()
    dims = Dict{Symbol, Int}()

    # 根据坐标系统确定维度键
    dim_keys = coordinate_system == :cartesian ? (:x, :y, :z) :
               coordinate_system == :cylindrical ? (:r, :θ, :z) :
               coordinate_system == :spherical ? (:r, :θ, :φ) :
               error("Unsupported coordinate system: $coordinate_system")
    
    for d in dim_keys
        lower, upper = limits[d]
        # 处理网格细化、拉伸或均匀网格
        if haskey(adaptive_params, d)
            # 自适应网格的生成逻辑
            # 省略具体细节，参考原代码逻辑
        elseif haskey(stretch_funcs, d)
            # 网格拉伸
        else
            # 均匀网格
        end
    end
    
    # 初始化物理场与配置为空字典
    physical_fields = Dict{Symbol, Any}()
    config = Dict{Symbol, Any}()

    return Grid(coordinate_system, limits, spacing, coordinates, dims, bc,
                adaptive_params, custom_bc, physical_fields, config, refinement_level)
end

##########################################################################
# 网格细化相关函数
##########################################################################

"""
    update_refinement_level!(grid::Grid, new_level::Int)

根据网格细化级别的更新，调整网格的细化级别，并通知其他模块更新相关参数（如EOS）。
"""
function update_refinement_level!(grid::Grid, new_level::Int)
    grid.refinement_level = new_level
    println("更新网格细化级别至: $new_level")
    
    # 根据新的细化级别，动态调整EOS等物理模型
    adaptive_eos_coupling(grid.coordinates[:r], grid.physical_fields[:temperature], :default, grid.refinement_level)
end

"""
    get_refinement_level(grid::Grid) -> Int

获取当前网格的细化级别。
"""
function get_refinement_level(grid::Grid)
    return grid.refinement_level
end

# --------------------------
# 网格细化函数
# --------------------------

"""
    refine_grid_combined!(amr::AdaptiveMeshRefinement, field::Symbol, gradient_threshold::Float64, value_threshold::Float64, eos::FiniteTempEOS)

结合梯度和物理场值进行网格细化，在两个标准都满足的情况下细化网格。
"""
function refine_grid_combined!(amr::AdaptiveMeshRefinement, field::Symbol, gradient_threshold::Float64, value_threshold::Float64, eos::FiniteTempEOS)
    gradient = compute_gradient(amr, field)
    field_data = amr.physical_fields[field]

    for i in 1:length(gradient)
        if gradient[i] > gradient_threshold && field_data[i] > value_threshold
            # 对梯度和物理场值都满足条件的区域进行细化
            amr.grid_size += 1
            amr.current_refinement_level = min(amr.current_refinement_level + 1, amr.max_refinement_level)
            println("在第 $(i) 位置细化网格，梯度和物理场值均满足条件")
        elseif gradient[i] < gradient_threshold / 2 && field_data[i] < value_threshold / 2
            # 对梯度和物理场值均较小的区域进行粗化
            amr.grid_size = max(amr.grid_size - 1, amr.min_refinement_level)
            amr.current_refinement_level = max(amr.current_refinement_level - 1, amr.min_refinement_level)
            println("在第 $(i) 位置粗化网格，梯度和物理场值均较低")
        end
    end

    # 根据当前网格细化级别调整EOS
    adaptive_eos_coupling(amr, eos)
end

"""
    refine_grid_by_value!(amr::AdaptiveMeshRefinement, field::Symbol, threshold::Float64, eos::FiniteTempEOS)

根据物理场的值自动细化网格，依据物理量的绝对值进行细化。
"""
function refine_grid_by_value!(amr::AdaptiveMeshRefinement, field::Symbol, threshold::Float64, eos::FiniteTempEOS)
    field_data = amr.physical_fields[field]

    for i in 1:length(field_data)
        if field_data[i] > threshold
            # 对物理场值超过阈值的区域增加网格密度
            amr.grid_size += 1
            amr.current_refinement_level = min(amr.current_refinement_level + 1, amr.max_refinement_level)
            println("在第 $(i) 位置细化网格，物理场值超过阈值")
        elseif field_data[i] < threshold / 2
            # 对物理场值较低的区域减少网格密度
            amr.grid_size = max(amr.grid_size - 1, amr.min_refinement_level)
            amr.current_refinement_level = max(amr.current_refinement_level - 1, amr.min_refinement_level)
            println("在第 $(i) 位置粗化网格，物理场值低于阈值")
        end
    end

    # 根据当前网格细化级别调整EOS
    adaptive_eos_coupling(amr, eos)
end

end  # module GridModule
