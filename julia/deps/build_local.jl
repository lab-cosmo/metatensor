using Pkg
# save the currently active project path to use later when saving preferences
project_path = dirname(Pkg.project().path)

# Install build time depenedencies
Pkg.activate(@__DIR__)
Pkg.instantiate()


using Scratch, Preferences, CMake_jll

Metatensor = Base.UUID("afba2530-6023-4bf7-b247-2c594027d0bd")

# get scratch directories
build_dir = get_scratch!(Metatensor, "build")
isdir(build_dir) && rm(build_dir; recursive=true)

install_dir = get_scratch!(Metatensor, "usr")
isdir(install_dir) && rm(install_dir; recursive=true)

source_dir = realpath(joinpath(@__DIR__, "..", "..", "metatensor-core"))
isdir(source_dir) || error("could not find metatensore-core sources")

# build and install
@info "Building metatensor from sources" source_dir build_dir install_dir
cmake() do cmake_path
    cmake_opts = [
        "-DBUILD_SHARED_LIBS=ON",
        "-DMETATENSOR_INSTALL_BOTH_STATIC_SHARED=OFF",
        "-DCMAKE_INSTALL_PREFIX=$(install_dir)",
    ]
    cmd = `$cmake_path $cmake_opts -B$(build_dir) -S$(source_dir)`
    @info "Configuring CMake" cmd
    run(cmd)

    cmd = `$cmake_path --build $(build_dir) --parallel --target install`
    @info "Building with CMake" cmd
    run(cmd)
end

possible_paths = [
    joinpath(install_dir, "lib", "libmetatensor.so"),
    joinpath(install_dir, "lib", "libmetatensor.dylib"),
    joinpath(install_dir, "bin", "metatensor.dll"),
]

lib_path = missing
for path in possible_paths
    if isfile(path)
        global lib_path
        lib_path = path
        break
    end
end

if ismissing(lib_path)
    error("could not find metatensor library in install directory")
end

# Save the path to the freshly built library in preferences, it will be checked
# when loading Metatensor.
@info "Setting preference to use local build in $(joinpath(project_path, "LocalPreferences.toml"))"
set_preferences!(
    joinpath(project_path, "LocalPreferences.toml"),
    "Metatensor",
    "libmetatensor" => lib_path;
    force=true
)
