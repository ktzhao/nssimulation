# GridModule.jl
#
# 本模块定义了用于物理模拟的网格数据结构与网格生成函数，
# 除了支持多种坐标系统、自适应网格（AMR）和网格拉伸功能外，
# 还扩展了物理场数据接口，支持初始化、更新及后续求解器调用，
# 同时提供从外部文件（HDF5、TOML、YAML）读取网格初始条件和配置参数的功能。
#
# 依赖：LinearAlgebra, HDF5, TOML, YAML
#
# 请确保已安装 HDF5.jl 与 YAML.jl （TOML 为 Julia 内置库）。

module GridModule

using LinearAlgebra
using HDF5
using TOML
using YAML

export Grid, create_grid, apply_boundary_conditions,
       init_physical_fields!, update_physical_field!,
       read_config

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
    physical_fields::Dict{Symbol,Any}
    config::Dict{Symbol,Any}
end

##########################################################################
# 网格生成函数
##########################################################################

"""
    create_grid(; coordinate_system, limits, spacing, bc, adaptive_params, stretch_funcs, custom_bc)

生成 Grid 对象。

关键字参数：
- coordinate_system::Symbol = :cartesian  
    坐标系统，支持 :cartesian、:cylindrical、:spherical。
- limits::Dict{Symbol,Tuple{Float64,Float64}}  
    各方向的域边界。笛卡尔坐标下应包含键 :x, :y, :z；
    柱坐标下包含 :r, :θ, :z；
    球坐标下包含 :r, :θ, :φ。
- spacing::Dict{Symbol,Float64}  
    各方向的基础网格间距。
- bc::Dict{Symbol,Any} = Dict{Symbol,Any}()  
    边界条件字典，如 :xlow => :Dirichlet、:xhigh => (:Neumann, nothing)、:ylow => :Absorbing、:zhigh => :Periodic 等。
- adaptive_params::Dict{Symbol,Tuple{Float64,Float64,Float64,Float64}} = Dict{Symbol,Tuple{Float64,Float64,Float64,Float64}}()  
    自适应网格参数，对于某一方向采用局部细化。
- stretch_funcs::Dict{Symbol, Function} = Dict{Symbol,Function}()  
    网格拉伸函数，若存在则生成非均匀网格。
- custom_bc::Dict{Symbol,Function} = Dict{Symbol,Function}()  
    用户自定义边界处理函数。
    
初始化时，物理场为空，配置参数为空。

返回值：Grid 对象。
"""
function create_grid(; coordinate_system::Symbol = :cartesian,
                     limits::Dict{Symbol, Tuple{Float64, Float64}},
                     spacing::Dict{Symbol, Float64},
                     bc::Dict{Symbol,Any} = Dict{Symbol,Any}(),
                     adaptive_params::Dict{Symbol, Tuple{Float64, Float64, Float64, Float64}} = Dict{Symbol,Tuple{Float64,Float64,Float64,Float64}}(),
                     stretch_funcs::Dict{Symbol, Function} = Dict{Symbol,Function}(),
                     custom_bc::Dict{Symbol, Function} = Dict{Symbol,Function}())
    coordinates = Dict{Symbol, Vector{Float64}}()
    dims = Dict{Symbol, Int}()
    
    # 根据坐标系统确定维度键
    dim_keys = coordinate_system == :cartesian ? (:x, :y, :z) :
               coordinate_system == :cylindrical ? (:r, :θ, :z) :
               coordinate_system == :spherical ? (:r, :θ, :φ) :
               error("Unsupported coordinate system: $coordinate_system")
    
    for d in dim_keys
        lower, upper = limits[d]
        # 若提供自适应参数，则采用三段式网格划分
        if haskey(adaptive_params, d)
            refine_start, refine_end, fine_spacing, coarse_spacing = adaptive_params[d]
            pts = Float64[]
            # 第一段：粗网格
            if lower < refine_start
                pts1 = collect(lower:coarse_spacing:refine_start)
                append!(pts, pts1)
            else
                push!(pts, lower)
            end
            # 第二段：细网格
            if refine_start < refine_end
                pts2 = collect(refine_start:fine_spacing:refine_end)
                if !isempty(pts) && isapprox(first(pts2), pts[end]; atol=1e-8)
                    pts2 = pts2[2:end]
                end
                append!(pts, pts2)
            end
            # 第三段：粗网格
            if refine_end < upper
                pts3 = collect(refine_end:coarse_spacing:upper)
                if !isempty(pts) && isapprox(first(pts3), pts[end]; atol=1e-8)
                    pts3 = pts3[2:end]
                end
                append!(pts, pts3)
            end
            if abs(pts[end] - upper) > 1e-8
                push!(pts, upper)
            end
            coordinates[d] = pts
            dims[d] = length(pts)
        elseif haskey(stretch_funcs, d)
            # 使用网格拉伸生成非均匀网格
            n = Int(floor((upper - lower) / spacing[d])) + 1
            u_vals = collect(range(0.0, stop=1.0, length=n))
            stretched = [stretch_funcs[d](u) for u in u_vals]
            coordinates[d] = stretched
            dims[d] = length(stretched)
        else
            # 默认均匀网格
            n = Int(floor((upper - lower) / spacing[d])) + 1
            coordinates[d] = collect(range(lower, stop=upper, length=n))
            dims[d] = n
        end
    end
    
    # 初始化物理场与配置为空字典
    physical_fields = Dict{Symbol,Any}()
    config = Dict{Symbol,Any}()
    
    return Grid(coordinate_system, limits, spacing, coordinates, dims, bc,
                adaptive_params, custom_bc, physical_fields, config)
end

##########################################################################
# 边界条件应用函数
##########################################################################

"""
    apply_boundary_conditions(field::Array{Float64,1}, boundary::Symbol, grid::Grid)

对一维场变量 field 应用边界条件。支持预定义边界条件类型：
- :Dirichlet  —— 若给出数值，则直接赋值；
- :Neumann    —— 采用相邻内部值（零梯度）；
- :Absorbing  —— 将边界设置为 0（或衰减）；
- :Periodic  —— 采用周期性边界，从对侧复制数据；
- :Mixed     —— 混合边界，示例中采用简单平均（用户可通过 custom_bc 自定义）。
若 grid.custom_bc 中为该边界键定义了函数，则优先使用自定义函数。

返回修改后的 field。
"""
function apply_boundary_conditions(field::Array{Float64,1}, boundary::Symbol, grid::Grid)
    if haskey(grid.custom_bc, boundary)
        return grid.custom_bc[boundary](field)
    end
    if !haskey(grid.bc, boundary)
        return field
    end
    bc_setting = grid.bc[boundary]
    if bc_setting isa Tuple
        bc_type, bc_value = bc_setting
    else
        bc_type = bc_setting
        bc_value = nothing
    end

    if bc_type == :Dirichlet && bc_value !== nothing
        field[1] = bc_value
    elseif bc_type == :Neumann
        field[1] = field[2]
    elseif bc_type == :Absorbing
        field[1] = 0.0
    elseif bc_type == :Periodic
        field[1] = field[end-1]
    elseif bc_type == :Mixed && bc_value !== nothing
        field[1] = 0.5*(field[2] + bc_value)
    end
    return field
end

##########################################################################
# 物理场数据接口：初始化与更新
##########################################################################

"""
    init_physical_fields!(grid::Grid, field_data::Dict{Symbol, T}) where T

初始化 Grid 中的物理场数据。field_data 的键（如 :density, :pressure, :magnetic_field, :electric_field）
对应的值通常为数值数组或其它数据结构，由后续求解器调用使用。

此函数将 field_data 合并到 grid.physical_fields 中（若存在同名字段则覆盖）。

返回修改后的 grid（就地修改）。
"""
function init_physical_fields!(grid::Grid, field_data::Dict{Symbol, T}) where T
    for (key, value) in field_data
        if !haskey(grid.physical_fields, key)
            error("物理场 $key 在 grid.physical_fields 中不存在。")
        end
        if typeof(value) != typeof(grid.physical_fields[key])
            error("物理场 $key 的类型不匹配：预期 $(typeof(grid.physical_fields[key])), 实际 $(typeof(value))。")
        end
        grid.physical_fields[key] = value
    end
    return grid
end

"""
    update_physical_field!(grid::Grid, field_name::Symbol, new_data)

更新 Grid 中指定的物理场数据为 new_data。

返回修改后的 grid（就地修改）。
"""
function update_physical_field!(grid::Grid, field_name::Symbol, new_data)
    grid.physical_fields[field_name] = new_data
    return grid
end

##########################################################################
# 配置参数读取接口
##########################################################################

"""
    read_config(filename::String) -> Dict{Symbol,Any}

从外部文件中读取配置参数，支持 TOML、YAML 和 HDF5 文件格式，
根据文件扩展名自动选择解析方法。

返回值为配置参数字典，所有键均转换为 Symbol（递归转换）。
"""
# 辅助函数：递归地将字典中的键转换为 Symbol
function convert_keys_to_symbols(x)
    if x isa Dict
        return Dict(Symbol(k) => convert_keys_to_symbols(v) for (k, v) in x)
    elseif x isa Array
        return [convert_keys_to_symbols(elem) for elem in x]
    else
        return x
    end
end

function read_config(filename::String)::Dict{Symbol,Any}
    if endswith(lowercase(filename), ".toml")
        config_str = read(filename, String)
        config_dict = TOML.parse(config_str)
    elseif endswith(lowercase(filename), ".yaml") || endswith(lowercase(filename), ".yml")
        config_dict = YAML.load_file(filename)
    elseif endswith(lowercase(filename), ".h5") || endswith(lowercase(filename), ".hdf5")
        config_dict = Dict{String,Any}()
        h5file = h5open(filename, "r")
        # 假设配置存储在根组的属性中
        for attr in attributes(h5file)
            config_dict[attr] = read_attribute(h5file, attr)
        end
        close(h5file)
    else
        error("不支持的配置文件格式: $filename")
    end
    # 递归转换所有键为 Symbol
    return convert_keys_to_symbols(config_dict)
end

end  # module GridModule
