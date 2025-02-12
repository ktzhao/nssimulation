module Communication

using LinearAlgebra
using Main.GridModule
using Main.GRMHDModule

export exchange_boundaries, gather_data, scatter_data, communicate_fields

# --------------------------
# 数据交换函数
# --------------------------

"""
    exchange_boundaries(grid::Grid)

该函数用于交换网格边界上的数据，确保每个计算单元之间的信息流动。
"""
function exchange_boundaries(grid::Grid)
    # Exchange the left and right boundaries
    for i in 1:length(grid.coordinates[:y])
        # Left boundary with right boundary
        grid.physical_fields[:density][1, i] = grid.physical_fields[:density][end, i]
        grid.physical_fields[:temperature][1, i] = grid.physical_fields[:temperature][end, i]
        grid.physical_fields[:pressure][1, i] = grid.physical_fields[:pressure][end, i]
        grid.physical_fields[:magnetic_field][1, i] = grid.physical_fields[:magnetic_field][end, i]
        grid.physical_fields[:current_density][1, i] = grid.physical_fields[:current_density][end, i]

        grid.physical_fields[:density][end, i] = grid.physical_fields[:density][1, i]
        grid.physical_fields[:temperature][end, i] = grid.physical_fields[:temperature][1, i]
        grid.physical_fields[:pressure][end, i] = grid.physical_fields[:pressure][1, i]
        grid.physical_fields[:magnetic_field][end, i] = grid.physical_fields[:magnetic_field][1, i]
        grid.physical_fields[:current_density][end, i] = grid.physical_fields[:current_density][1, i]
    end
    # Exchange the top and bottom boundaries
    for i in 1:length(grid.coordinates[:x])
        grid.physical_fields[:density][i, 1] = grid.physical_fields[:density][i, end]
        grid.physical_fields[:temperature][i, 1] = grid.physical_fields[:temperature][i, end]
        grid.physical_fields[:pressure][i, 1] = grid.physical_fields[:pressure][i, end]
        grid.physical_fields[:magnetic_field][i, 1] = grid.physical_fields[:magnetic_field][i, end]
        grid.physical_fields[:current_density][i, 1] = grid.physical_fields[:current_density][i, end]

        grid.physical_fields[:density][i, end] = grid.physical_fields[:density][i, 1]
        grid.physical_fields[:temperature][i, end] = grid.physical_fields[:temperature][i, 1]
        grid.physical_fields[:pressure][i, end] = grid.physical_fields[:pressure][i, 1]
        grid.physical_fields[:magnetic_field][i, end] = grid.physical_fields[:magnetic_field][i, 1]
        grid.physical_fields[:current_density][i, end] = grid.physical_fields[:current_density][i, 1]
    end
end

# --------------------------
# 数据收集与分发
# --------------------------

"""
    gather_data(grid::Grid, data::Dict, fields::Vector{Symbol})

将指定的物理字段收集到数据字典中，供后续计算使用。
"""
function gather_data(grid::Grid, data::Dict, fields::Vector{Symbol})
    for field in fields
        data[field] = grid.physical_fields[field]
    end
end

"""
    scatter_data!(grid::Grid, data::Dict, fields::Vector{Symbol})

将计算结果分发到网格的物理场中。
"""
function scatter_data!(grid::Grid, data::Dict, fields::Vector{Symbol})
    for field in fields
        grid.physical_fields[field] .= data[field]
    end
end

# --------------------------
# 通信与数据传输
# --------------------------

"""
    communicate_fields(grid::Grid, eos::FiniteTempEOS)

该函数用来进行网格间的通信，确保各计算单元的物理量保持一致。
"""
function communicate_fields(grid::Grid, eos::FiniteTempEOS)
    # 交换物理场的边界数据
    exchange_boundaries(grid)

    # 传递新的物理量数据到计算单元
    # 例如温度、密度等
    gather_data(grid, grid.communication_data, [:density, :temperature, :pressure, :magnetic_field, :current_density])

    # 对数据进行处理，优化计算
    # 如需要，可以在这里进行物理场的滤波或平滑
    process_data!(grid)

    # 将更新后的物理场数据分发回网格
    scatter_data!(grid, grid.communication_data, [:density, :temperature, :pressure, :magnetic_field, :current_density])
end

"""
    process_data!(grid::Grid)

对数据进行处理，例如物理场的滤波、平滑或其他数据优化步骤
"""
function process_data!(grid::Grid)
    # 可以加入对温度、压力或其他字段的平滑处理
    for i in 2:(length(grid.coordinates[:x])-1)
        for j in 2:(length(grid.coordinates[:y])-1)
            # 平滑操作，例如计算邻域平均
            grid.physical_fields[:density][i, j] = mean([grid.physical_fields[:density][i, j], 
                                                         grid.physical_fields[:density][i-1, j], 
                                                         grid.physical_fields[:density][i+1, j]])
            grid.physical_fields[:temperature][i, j] = mean([grid.physical_fields[:temperature][i, j], 
                                                            grid.physical_fields[:temperature][i-1, j], 
                                                            grid.physical_fields[:temperature][i+1, j]])
        end
    end
end

# --------------------------
# 数据同步与更新
# --------------------------

"""
    sync_fields(grid::Grid, eos::FiniteTempEOS)

同步网格上所有区域的物理场数据。
"""
function sync_fields(grid::Grid, eos::FiniteTempEOS)
    # 进行通信以确保各计算单元的数据同步
    communicate_fields(grid, eos)

    # 更新物理场数据
    for i in 1:length(grid.coordinates[:x])
        for j in 1:length(grid.coordinates[:y])
            # 更新每个位置的密度、温度、压力等物理量
            rho = grid.physical_fields[:density][i, j]
            T = grid.physical_fields[:temperature][i, j]
            P = eos.pressure(rho, T)
            grid.physical_fields[:pressure][i, j] = P
        end
    end
end

end # module Communication
