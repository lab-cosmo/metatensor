import glob
import os
import subprocess
import sys

import packaging.version
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel


ROOT = os.path.realpath(os.path.dirname(__file__))

METATENSOR_BUILD_TYPE = os.environ.get("METATENSOR_BUILD_TYPE", "release")
if METATENSOR_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{METATENSOR_BUILD_TYPE}', "
        "expected 'debug' or 'release'"
    )

METATENSOR_CORE = os.path.join(ROOT, "..", "..", "metatensor-core")


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.7. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """
    Build the native library using cmake
    """

    def run(self):
        source_dir = METATENSOR_CORE
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "metatensor")

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={METATENSOR_BUILD_TYPE}",
            "-DBUILD_SHARED_LIBS=ON",
            "-DMETATENSOR_INSTALL_BOTH_STATIC_SHARED=OFF",
            # strip dynamic library for smaller wheels to download/install
            "-DEXTRA_RUST_FLAGS=-Cstrip=symbols",
        ]

        if "CARGO" in os.environ:
            cmake_options.append(f"-DCARGO_EXE={os.environ['CARGO']}")

        # Handle cross-compilation by detecting cibuildwheels environnement
        # variables
        if sys.platform.startswith("darwin"):
            # ARCHFLAGS is set by cibuildwheels
            ARCHFLAGS = os.environ.get("ARCHFLAGS")
            if ARCHFLAGS is not None:
                archs = filter(
                    lambda u: bool(u),
                    ARCHFLAGS.strip().split("-arch "),
                )
                archs = list(archs)
                assert len(archs) == 1
                arch = archs[0].strip()

                if arch == "x86_64":
                    cmake_options.append("-DRUST_BUILD_TARGET=x86_64-apple-darwin")
                elif arch == "arm64":
                    cmake_options.append("-DRUST_BUILD_TARGET=aarch64-apple-darwin")
                else:
                    raise ValueError(f"unknown arch: {arch}")

        elif sys.platform.startswith("linux"):
            # we set RUST_BUILD_TARGET in our custom docker image
            RUST_BUILD_TARGET = os.environ.get("RUST_BUILD_TARGET")
            if RUST_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={RUST_BUILD_TARGET}")

        elif sys.platform.startswith("win32"):
            # CARGO_BUILD_TARGET is set by cibuildwheels
            CARGO_BUILD_TARGET = os.environ.get("CARGO_BUILD_TARGET")
            if CARGO_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={CARGO_BUILD_TARGET}")

        else:
            raise ValueError(f"unknown platform: {sys.platform}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--parallel", "--target", "install"],
            check=True,
        )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs.\nUse `pip install .` or "
            "`python -m build --wheel . && pip install "
            "dist/metatensor_core-*.whl` to install from source."
        )


class sdist_generate_data(sdist):
    """
    Create a sdist with an additional generated files:
        - `n_commits_since_last_tag`
        - `metatensor-core-cxx-*.tar.gz`
    """

    def run(self):
        with open("n_commits_since_last_tag", "w") as fd:
            fd.write(str(n_commits_since_last_tag()))

        generate_cxx_tar()

        # run original sdist
        super().run()

        os.unlink("n_commits_since_last_tag")
        for path in glob.glob("metatensor-core-cxx-*.tar.gz"):
            os.unlink(path)


def generate_cxx_tar():
    script = os.path.join(ROOT, "..", "..", "scripts", "package-core.sh")
    assert os.path.exists(script)

    try:
        output = subprocess.run(
            ["bash", "--version"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
    except Exception as e:
        raise RuntimeError("could not run `bash`, is it installed?") from e

    stderr = ""
    stdout = ""

    output = subprocess.run(
        ["bash", script, os.getcwd()],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf8",
    )
    if output.returncode != 0:
        stderr = output.stderr
        stdout = output.stdout
        raise RuntimeError(
            "failed to collect C++ sources for Python sdist\n"
            f"stdout:\n {stdout}\n\nstderr:\n {stderr}"
        )


def get_rust_version():
    # read version from Cargo.toml
    with open(os.path.join(METATENSOR_CORE, "Cargo.toml")) as fd:
        for line in fd:
            if line.startswith("version"):
                _, version = line.split(" = ")
                # remove quotes
                version = version[1:-2]
                # take the first version in the file, this should be the right
                # version
                break

    return version


def n_commits_since_last_tag():
    """
    If git is available and we are building from a checkout, get the number of commits
    since the last tag. Otherwise, this always returns 0.
    """
    script = os.path.join(ROOT, "..", "..", "scripts", "n-commits-since-last-tag.py")
    assert os.path.exists(script)

    TAG_PREFIX = "metatensor-core-v"
    output = subprocess.run(
        [sys.executable, script, TAG_PREFIX],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf8",
    )

    if output.returncode != 0:
        raise Exception(
            "failed to get number of commits since last tag.\n"
            f"stdout: {output.stdout}\n"
            f"stderr: {output.stderr}\n"
        )
    elif output.stderr:
        print(output.stderr, file=sys.stderr)
        return 0
    else:
        return int(output.stdout)


def create_version_number(version):
    version = packaging.version.parse(version)

    if os.path.exists("n_commits_since_last_tag"):
        # we are building from a sdist, without git available, but the git
        # version was recorded in the `n_commits_since_last_tag` file
        with open("n_commits_since_last_tag") as fd:
            n_commits = int(fd.read().strip())
    else:
        n_commits = n_commits_since_last_tag()

    if n_commits != 0:
        # if we have commits since the last tag, this mean we are in a pre-release of
        # the next version. So we increase either the minor version number or the
        # release candidate number (if we are closing up on a release)
        if version.pre is not None:
            assert version.pre[0] == "rc"
            pre = ("rc", version.pre[1] + 1)
            release = version.release
        else:
            major, minor, patch = version.release
            release = (major, minor + 1, 0)
            pre = None

        # this is using a private API which is intended to become public soon:
        # https://github.com/pypa/packaging/pull/698. In the mean time we'll
        # use this
        version._version = version._version._replace(release=release)
        version._version = version._version._replace(pre=pre)
        version._version = version._version._replace(dev=("dev", n_commits))

    return str(version)


if __name__ == "__main__":
    if not os.path.exists(METATENSOR_CORE):
        # we are building from a sdist, which should include metatensor-core Rust
        # sources as a tarball
        tarballs = glob.glob(os.path.join(ROOT, "metatensor-core-*.tar.gz"))

        if not len(tarballs) == 1:
            raise RuntimeError(
                "expected a single 'metatensor-core-*.tar.gz' file containing "
                "metatensor-core Rust sources. remove all files and re-run "
                "scripts/package-core.sh"
            )

        METATENSOR_CORE = os.path.realpath(tarballs[0])
        subprocess.run(
            ["cmake", "-E", "tar", "xf", METATENSOR_CORE],
            cwd=ROOT,
            check=True,
        )

        METATENSOR_CORE = ".".join(METATENSOR_CORE.split(".")[:-2])

    with open(os.path.join(ROOT, "AUTHORS")) as fd:
        authors = fd.read().splitlines()

    if authors[0].startswith(".."):
        # handle "raw" symlink files (on Windows or from full repo tarball)
        with open(os.path.join(ROOT, authors[0])) as fd:
            authors = fd.read().splitlines()

    setup(
        version=create_version_number(get_rust_version()),
        author=", ".join(authors),
        ext_modules=[
            # only declare the extension, it is built & copied as required by cmake
            # in the build_ext command
            Extension(name="metatensor", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
            "sdist": sdist_generate_data,
        },
        package_data={
            "metatensor-core": [
                "metatensor/core/lib/*",
                "metatensor/core/include/*",
            ]
        },
    )
