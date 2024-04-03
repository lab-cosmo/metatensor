import os
import subprocess
import sys
import uuid

import packaging.version
from setuptools import setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.sdist import sdist


ROOT = os.path.realpath(os.path.dirname(__file__))
METATENSOR_CORE = os.path.realpath(os.path.join(ROOT, "..", "metatensor-core"))

METATENSOR_OPERATIONS_VERSION = "0.2.1"


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs.\nUse `pip install .` or "
            "`python -m build --wheel . && pip install "
            "dist/metatensor_operations-*.whl` to install from source."
        )


class sdist_git_version(sdist):
    """
    Create a sdist with an additional generated file containing the extra
    version from git.
    """

    def run(self):
        with open("n_commits_since_last_tag", "w") as fd:
            fd.write(str(n_commits_since_last_tag()))

        # run original sdist
        super().run()

        os.unlink("n_commits_since_last_tag")


def n_commits_since_last_tag():
    """
    If git is available and we are building from a checkout, get the number of commits
    since the last tag. Otherwise, this always returns 0.
    """
    script = os.path.join(ROOT, "..", "..", "scripts", "n-commits-since-last-tag.py")
    assert os.path.exists(script)

    TAG_PREFIX = "metatensor-operations-v"
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
    with open(os.path.join(ROOT, "AUTHORS")) as fd:
        authors = fd.read().splitlines()

    if authors[0].startswith(".."):
        # handle "raw" symlink files (on Windows or from full repo tarball)
        with open(os.path.join(ROOT, authors[0])) as fd:
            authors = fd.read().splitlines()

    install_requires = []

    # when packaging a sdist for release, we should never use local dependencies
    METATENSOR_NO_LOCAL_DEPS = os.environ.get("METATENSOR_NO_LOCAL_DEPS", "0") == "1"

    if not METATENSOR_NO_LOCAL_DEPS and os.path.exists(METATENSOR_CORE):
        # we are building from a git checkout or full repo archive

        # add a random uuid to the file url to prevent pip from using a cached
        # wheel for metatensor-core, and force it to re-build from scratch
        uuid = uuid.uuid4()
        install_requires.append(f"metatensor-core @ file://{METATENSOR_CORE}?{uuid}")
    else:
        # we are building from a sdist/installing from a wheel
        install_requires.append("metatensor-core >=0.1.0,<0.2.0")

    setup(
        version=create_version_number(METATENSOR_OPERATIONS_VERSION),
        author=", ".join(authors),
        install_requires=install_requires,
        cmdclass={
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "sdist": sdist_git_version,
        },
    )
