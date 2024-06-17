import os
import subprocess
import sys
from datetime import datetime

from sphinx.domains.c import CObject


# When importing metatensor-torch, this will change the definition of the classes
# to include the documentation
os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"

import metatensor  # noqa: E402
import metatensor.torch  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "docs", "extensions"))


# We use a second (pseudo) sphinx project located in `docs/generate_examples` to run the
# examples and generate the actual output for our shinx-gallery. This is necessary
# because here we have to set `METATENSOR_IMPORT_FOR_SPHINX` to `"1"` allowing the
# correct generation of the class and function docstrings which are seperate from the
# actual code.
#
# We register and use the same sphinx gallery configuration as in the pseudo project.
sys.path.append(os.path.join(ROOT, "docs"))
from generate_examples.conf import sphinx_gallery_conf  # noqa


# ------------------------------------------------------------------------------------ #
# Monkey patching the C domain in sphinx to include functions in the page-local TOC.
# This is inspired by similar code in the C++ domain


def _object_hierarchy_parts(self, sig_node):
    names = self.env.temp_data["c:last_symbol"].get_full_nested_name().names
    return tuple(map(str, names))


def _toc_entry_name(self, sig_node):
    if not sig_node.get("_toc_parts"):
        return ""

    config = self.env.app.config
    objtype = sig_node.parent.get("objtype")
    if config.add_function_parentheses and objtype in {"function", "method"}:
        parens = "()"
    else:
        parens = ""

    *parents, name = sig_node["_toc_parts"]
    if config.toc_object_entries_show_parents == "domain":
        return ".".join((*self.env.temp_data.get("c:domain_name", ()), name + parens))
    if config.toc_object_entries_show_parents == "hide":
        return name + parens
    if config.toc_object_entries_show_parents == "all":
        return ".".join(parents + [name + parens])

    return ""


CObject._object_hierarchy_parts = _object_hierarchy_parts
CObject._toc_entry_name = _toc_entry_name

# End of monkey-patching code
# ------------------------------------------------------------------------------------ #


# -- Project information -----------------------------------------------------

project = "metatensor"
author = ", ".join(open(os.path.join(ROOT, "AUTHORS")).read().splitlines())
copyright = f"{datetime.now().date().year}, {author}"


def build_doxygen_docs():
    try:
        os.mkdir(os.path.join(ROOT, "docs", "build"))
    except OSError:
        pass

    # we need to run a cargo build to make sure the header is up to date
    subprocess.run(["cargo", "build"])
    subprocess.run(["doxygen", "Doxyfile"], cwd=os.path.join(ROOT, "docs"))


def generate_examples():
    # we can not run sphinx-gallery in the same process as the normal sphinx, since they
    # need to import metatensor.torch differently (with and without
    # METATENSOR_IMPORT_FOR_SPHINX=1). So instead we run it inside a small script, and
    # include the corresponding output later.
    del os.environ["METATENSOR_IMPORT_FOR_SPHINX"]
    script = os.path.join(ROOT, "docs", "generate_examples", "generate-examples.py")
    subprocess.run([sys.executable, script], capture_output=False)
    os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"


def setup(app):
    build_doxygen_docs()
    generate_examples()

    app.add_css_file("css/metatensor.css")


rst_prolog = f"""
.. |metatensor-core-version| replace:: {metatensor.__version__}
.. |metatensor-torch-version| replace:: {metatensor.torch.__version__}
.. |metatensor-operations-version| replace:: {metatensor.operations.__version__}
.. |metatensor-learn-version| replace:: {metatensor.learn.__version__}

.. |C-32x32| image:: /../static/images/logo-c.*
    :width: 32px
    :height: 32px
    :alt: C

.. |C-16x16| image:: /../static/images/logo-c.*
    :width: 16px
    :height: 16px
    :alt: C

.. |Cxx-32x32| image:: /../static/images/logo-cxx.*
    :width: 32px
    :height: 32px
    :alt: C++

.. |Cxx-16x16| image:: /../static/images/logo-cxx.*
    :width: 16px
    :height: 16px
    :alt: C++

.. |Rust-32x32| image:: /../static/images/logo-rust.*
    :width: 32px
    :height: 32px
    :alt: Rust

.. |Rust-16x16| image:: /../static/images/logo-rust.*
    :width: 16px
    :height: 16px
    :alt: Rust

.. |Python-32x32| image:: /../static/images/logo-python.*
    :width: 32px
    :height: 32px
    :alt: Python

.. |Python-16x16| image:: /../static/images/logo-python.*
    :width: 16px
    :height: 16px
    :alt: Python
"""

# -- General configuration ---------------------------------------------------

needs_sphinx = "4.0.0"

python_use_unqualified_type_names = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # official extensions
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    # third party extensions
    "sphinxcontrib.details.directive",
    "sphinx_gallery.gen_gallery",
    "sphinx_toggleprompt",
    "breathe",
    "myst_parser",
    "sphinx_design",
    "chemiscope.sphinx",
    # local extensions
    "versions_list",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "examples/index.rst",
    "examples/sg_execution_times.rst",
    "sg_execution_times.rst",
]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

breathe_projects = {
    "metatensor": os.path.join(ROOT, "docs", "build", "doxygen", "xml"),
}
breathe_default_project = "metatensor"
breathe_domain_by_extension = {
    "h": "c",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "furo"

html_title = "Metatensor"
html_favicon = "../logo/metatensor-64.png"

html_theme_options = {
    "light_logo": "images/metatensor-horizontal.png",
    "dark_logo": "images/metatensor-horizontal-dark.png",
    "sidebar_hide_name": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../static"]

html_js_files = [
    os.path.join("js", "custom.js"),
    (
        "https://plausible.io/js/script.js",
        {"data-domain": "docs.metatensor.org", "defer": "defer"},
    ),
]
