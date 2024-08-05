.. _atomistic-overview:

Overview
========

In recent years, there has been an explosion in the use of machine learning for
atomistic applications, in particular using ML models as interatomic potentials
for molecular simulations (i.e. molecular dynamics, Monte Carlo, …). This, in
turn, creates a lot of redundant work integrating new ML models with simulation
engines (such as `ASE`_, `LAMMPS`_, `i-PI`_, `GROMACS`_, and many more!). Each
new ML model needs to write code to integrate each relevant engine one by one.

Our goal with metatensor atomistic models is to define a very clear boundary
between the model and the engines, such that models following the metatensor
interface can be used with **any** simulation engine (provided they know how to
use the interface); and that simulation engine only need to implement code to
use to ML models once, and get access to **all** machine learning models
following this interface.

.. figure:: /../static/images/goal-simulations.*
    :width: 500px
    :align: center

    Different steps in the workflow of running simulations with metatensor.
    Defining a model, training a model and running simulations with it can be
    done by different users; and the same metatensor-based model can be used
    with multiple simulation engines.

.. py:currentmodule:: metatensor.torch

This atomistic models interface is based on metatensor data format, and make
extensive use of data format ability to express sparsity (for example when
storing neighbor lists) and self-describing properties (to communicate what
exactly a model output contains in a generic manner). Using metatensor rich data
types (:py:class:`Labels`, :py:class:`TensorBlock`, and :py:class:`TensorMap`)
in the atomistic interface is what allows us to make the interface smaller and
with fewer special cases. The same interface can be used to communicate about
both complex (e.g. electron density, Hamiltonian matrix elements) and simple
(e.g. energy, atomic charges) predictions of the models; and support multiple
pathways for the prediction of gradients of properties.


.. seealso::

    We have a couple of :ref:`tutorials <atomistic-tutorials>` to learn how to
    define, export, and use metatensor atomistic models.

.. _ASE: https://wiki.fysik.dtu.dk/ase/ase/md.html
.. _LAMMPS: https://lammps.org/
.. _i-PI: https://ipi-code.org/
.. _GROMACS: https://www.gromacs.org/


Why use metatensor for atomistic models
---------------------------------------

The reason for using metatensor atomistic models will depend on which kind of
user you are (the same user can fall into multiple categories at different
points in time!):

Creating new machine learning architectures
    You are working to define new ML architectures, incorporating the latest ML
    research and coming up with new ideas to make ML models better.

    By using metatensor, you'll get to make your architecture available to
    everyone immediately and with fewer efforts. You'll also potentially get to
    delegate the work on the simulation engine interface to other developers, by
    sharing a single metatensor-based implementation with them.


Training existing architectures on new datasets
    You are taking existing architectures, and training them on your own
    dataset.

    By using metatensor, you'll get the ability to immediately test your model
    inside a Python environment (with Python-based simulation engines) and once
    you are confident with it, scale your simulations to larger scales while
    keeping the exact same model. You can also more easily integrate various
    architectures in your workflow (or even combine multiple models) and compare
    them for your own data.


Running simulations to study specific systems
    You want to study a specific system, and machine learning is only one of the
    tools in your toolbox. You might be training your own models, or using
    pre-trained models from someone else.

    By using metatensor, you'll get too use simulation software you are already
    familiar with, instead of having to install and learn new software just to
    use one specific ML model. You'll also get an easy way to try and compare
    existing models: just load them in your simulation engine and hit the floor
    running!


Developing of simulation engines
    You are working on software for molecular simulations, including algorithms
    to sample different thermodynamic ensembles, or high performance simulation
    code.

    By using metatensor, you'll get access to the whole space of machine
    learning potentials at once! You'll also get to use models for more than
    predicting the energy of a system (for example using ML models for charge
    transfers, predicting polarizability along a trajectory, *etc.*).

How it works
------------

.. py:currentmodule:: metatensor.torch.atomistic

Metatensor atomistic models are based on PyTorch, and more particularly
`TorchScript`_. TorchScript is a programming language which is mainly a subset
of Python, and PyTorch contains a compiler from Python to TorchScript code.
After doing this translation, the model no longer depends on Python and can be
executed directly inside simulation engines implemented in C, C++, Fortran, …
This approach allow to define and tweak models as Python code, and then once
they are working as intended, export them to a Python-independent representation
to be used in simulations.

In practice, models should be defined as custom :py:class:`torch.nn.Module`
sub-class, following our :py:class:`ModelInterface`. New models can be written
using this interface directly, and pre-existing models can use a small wrapper
to convert from this interface to the model's existing input and output data.
The models take as input a set of atomistic :py:class:`System` (typically a
single one during simulations, and multiple systems during training); a set of
``outputs`` requested by the engine, and should make prediction for all
properties in the ``outputs``. All predictions are then returned to the engine
in a dictionary of :py:class:`metatensor.torch.TensorMap`, one such tensor map
for each property (i.e. energy, atomic charges, dipole, electronic density,
chemical shielding, *etc.*)

Once a model is defined and trained, it should be exported by constructing a
:py:class:`MetatensorAtomisticModel`, and calling ``export`` on it. This class
is a wrapper for the model that will handle unit conversions on input and
outputs. It will also store metadata about the model (such as the authors, a
list of references, …) and the model capabilities (what properties it can
compute, which neighbors list the whole model requires, …). Optionally, this
class can also check that both data provided by the engine and properties
computed by the model follow the metatensor interface, which can be used to
debug your code.

Finally, the exported model can be loaded by simulation engines and used to run
simulations and make predictions.


.. _TorchScript: https://pytorch.org/docs/stable/jit.html

Constrains on atomistic models
------------------------------

There are a couple of constrains on what a given model must do to be useable
with metatensor, but apart from these you can do what you want inside the
model!

The main constrain is that the model must be compatible with `TorchScript`_,
i.e. you must use either pure PyTorch code in the definition of your model, or
implement a custom TorchScript extension for any operations where a pure PyTorch
implementation is too slow or too much work. See the `corresponding
documentation <torch-extensions>_` for more information on custom TorchScript
extensions.

.. _torch-extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html

Your model should also only take data from our :py:class:`System` definition:
atomic types and positions, simulation cell, and list of neighbors for different
spherical cutoffs.

If you need additional data that you can not compute inside the model (such as
atomic spins, non-spherical particle orientation, *etc.*) you can use
:py:meth:`System.get_data`, but this comes with significant caveats:

- anything going through :py:meth:`System.get_data` is experimental with no
  stability guarantee;
- you must modify the engine code to use :py:meth:`System.add_data` to add the
  required data to the systems;

If you need such data, please contact us (using email or `GitHub issues`_) to
formulate a plan to add it to metatensor interface!

.. _GitHub issues: https://github.com/lab-cosmo/metatensor/issues/new

Finally, your model can compute and output what it wants, and organize the data
and metadata of the outputs as it pleases, except for a set of standardized
outputs (identified by the corresponding key in the output dictionary). These
standardized outputs are documented in :ref:`this page
<atomistic-models-outputs>`.


.. _model-dataflow:

Data flow between the model and engine
--------------------------------------

The sequence of operations to use a metatensor atomistic model from a simulation
engine follows the same high level sequence of operations, illustrated and
explained below.


.. make the `tip` admonition grey only for this page
.. raw:: html

    <style>
        body[data-theme="light"] {
            --color-admonition-title--tip: #7c7c7c;
            --color-admonition-title-background--tip: #b9b9b9;
        }

        body[data-theme="auto"] {
            @media (prefers-color-scheme: light) {
                --color-admonition-title--tip: #7c7c7c;
                --color-admonition-title-background--tip: #b9b9b9;
            }
        }
    </style>

.. figure:: ../../static/images/model-dataflow.*
    :width: 600px
    :align: center

    Illustration of the flow of data between the engine and the model.

1. the engine loads an exported model from a file;

   .. tip::

        The engine should use :py:func:`check_atomistic_model` or
        :cpp:func:`metatensor_torch::load_atomistic_model` to also perform
        checks before loading the model.

2. the engine requests and gets its capabilities from the model;

   .. tip::

        This can be done by calling
        :py:func:`MetatensorAtomisticModel.capabilities`. This function is also
        exported to TorchScript and can be called from C++ with
        :cpp:func:`torch::jit::Module::run_method`.

3. the engine creates the :py:class:`ModelEvaluationOptions` based on the
   model's capabilities and user input;

4. the engine creates a list of :py:class:`System` (typically the list only
   contains one system) matching its own internal data representation;

   .. tip::
        The ``positions`` and ``cell`` should have their respective
        ``requires_grad`` parameters set if the engine wants to run backward
        propagation at step 10.

5. the engine asks the model for the required neighbor lists;

   .. tip::

        This can be done by calling
        :py:func:`MetatensorAtomisticModel.requested_neighbor_lists`. This
        function is also exported to TorchScript and can be called from C++ with
        :cpp:func:`torch::jit::Module::run_method`.

6. the engine computes the neighbor lists corresponding to the model requests,
   and register them with all systems;

   .. tip::

        If the engine does not use torch to compute the neighbor lists (using
        instead some other neighbors list implementation), the neighbors list
        should be registered with torch's automatic differentiation framework by
        using :py:func:`register_autograd_neighbors` before adding the neighbors
        lists to the systems.

        We provide a set of regression tests for neighbors lists in
        `metatensor-torch/tests/neighbor-checks`. The data in these files can be
        used to validate that a specific engine is computing the expected set of
        pairs for integration with metatensor models.

7. the engine calls the model ``forward()`` function with all the systems, the
   evaluations options and selected atoms, if any;
8. the model runs and executes its calculations;
9. the model returns all the requested outputs to the engine;
10. if needed, the engine runs ``backward()`` on the outputs to get gradients of
    some outputs with backward propagation;

    .. tip::

        The API for metatensor atomistic models also supports gradients computed
        during the forward pass with :py:attr:`ModelOutput.explicit_gradients`.
        Most models will not support this option though, and as such it is
        better to try to rely on backward differentiation gradients where
        possible.
