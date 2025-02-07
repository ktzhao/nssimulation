using .GridModule

# Define domain limits and grid spacing.
xmin, xmax, dx = 0.0, 320.0, 1.0
ymin, ymax, dy = 0.0, 320.0, 10.0
zmin, zmax, dz = 0.0, 320.0, 10.0

# Create the grid.
grid = create_grid(xmin, xmax, dx, ymin, ymax, dy, zmin, zmax, dz)

println("Grid created with dimensions: ", grid.nx, " x ", grid.ny, " x ", grid.nz)
println("x-coordinates: ", grid.x)

using Plots

# 选择 z 坐标上的一个固定层，例如 z = grid.z[round(Int, length(grid.z)/2)]
z_fixed = grid.z[round(Int, length(grid.z)/2)]
x_pts = grid.x
y_pts = grid.y

# 生成所有网格点的 x-y 坐标（二维散点图）
scatter(x_pts, y_pts, xlabel="x", ylabel="y", title="Grid Points on z = $z_fixed", legend=false)
savefig("grid_xy_slice.png")
println("2D grid slice plot saved as: grid_xy_slice.png")