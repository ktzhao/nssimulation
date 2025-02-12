module IOManager

using HDF5
using Distributed
using LinearAlgebra
#using FileLock
#using SharedVector
using Random

# 文件锁定机制，用于并行环境下避免多进程写入冲突
const lockfile = "lockfile.lock"

# 并行 I/O 模块：使用 HDF5 存储数据
function save_data_parallel(filename::String, data::Array, group_name::String)
    # 缓冲区写入
    buffer = SharedVector{Float64}(undef, length(data))
    
    # 使用并行存储方法存储数据
    @distributed for i in 1:length(data)
        buffer[i] = data[i]
    end

    # 文件锁定：防止多线程写入冲突
    lock = FileLock(lockfile)
    @lock lock begin
        h5open(filename, "a") do file
            # 将数据以并行方式批量写入文件
            dset = create_or_open(file, group_name, HDF5.H5T_NATIVE_DOUBLE, size(data))
            write(dset, buffer)
        end
    end
end

# 保存检查点：周期性保存模拟状态
function save_checkpoint(filename::String, state::Dict, chunk_size::Int=100)
    # 增量保存，避免每次都写入整个数据
    h5open(filename, "a") do file
        for (key, value) in state
            if !haskey(file, key)
                # 如果文件中没有这个数据集，则创建
                create_dataset(file, key, HDF5.H5T_NATIVE_DOUBLE, size(value))
            end
            
            # 增量保存：只保存与上次保存不同的部分
            dset = file[key]
            data_shape = size(value)
            start_idx = get_start_idx(dset, data_shape)
            
            # 将新增部分写入数据集
            write(dset, start_idx, value)
        end
    end
end

# 恢复检查点：从文件中恢复模拟状态
function load_checkpoint(filename::String)
    state = Dict()
    h5open(filename, "r") do file
        for key in keys(file)
            state[key] = read(file[key])
        end
    end
    return state
end

# 获取数据集的起始索引，用于增量保存
function get_start_idx(dset, data_shape)
    current_shape = size(dset)
    start_idx = current_shape[1] + 1
    if start_idx > data_shape[1]
        throw("数据超出存储范围")
    end
    return (start_idx, 1)  # 假设数据按行分块存储
end

# 创建或打开数据集
function create_or_open(file, group_name, datatype, shape)
    if haskey(file, group_name)
        return file[group_name]
    else
        return create_dataset(file, group_name, datatype, shape)
    end
end

# 异步 I/O 操作，使用多线程写入（增强性能）
function async_save_checkpoint(filename::String, state::Dict, chunk_size::Int=100)
    tasks = []
    for (key, value) in state
        push!(tasks, @spawn begin
            save_checkpoint(filename, Dict(key => value), chunk_size)
        end)
    end
    fetch.(tasks)  # 等待所有任务完成
end

end # module IOManager
