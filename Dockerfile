# 选择官方 Julia slim 版本作为基础镜像
FROM julia:1.9.3-slim

# 设置工作目录（后续所有操作均在此目录中执行）
WORKDIR /home/zhao/juliaworkspace

# 将项目配置文件复制到镜像中
# 如果你有 Manifest.toml，也一起复制；如果没有 Manifest.toml，这一行会匹配不到文件，可以忽略
COPY Project.toml Manifest.toml* ./

# 利用 Julia 的 Pkg.instantiate() 命令安装 Project.toml 中指定的所有依赖，
# 并预编译所有包以加快后续启动速度
RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# 安装 IJulia 包，并注册内核，这样 Binder 构建完成后，Notebook 就能使用“Julia”内核
RUN julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia; installkernel("Julia", "--project=/home/julia/project")'

# 暴露 Jupyter Notebook 默认使用的端口（Binder 会自动使用该端口）
EXPOSE 8888

# 设置默认启动命令，这里启动 Notebook 服务
CMD ["julia", "--project=/home/julia/project", "-e", "using IJulia; notebook()"]