include("GridModule.jl") 

using .GridModule

# Define domain limits and grid spacing.
xmin, xmax, dx = 0.0, 320.0, 10.0
ymin, ymax, dy = 0.0, 320.0, 10.0
zmin, zmax, dz = 0.0, 320.0, 10.0

# Create the grid.
grid = create_grid(xmin, xmax, dx, ymin, ymax, dy, zmin, zmax, dz)

println("Grid created with dimensions: ", grid.nx, " x ", grid.ny, " x ", grid.nz)
println("x-coordinates: ", grid.x)