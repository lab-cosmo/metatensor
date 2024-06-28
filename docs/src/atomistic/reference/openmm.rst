OpenMM integration
==================

.. py:currentmodule:: metatensor.torch.atomistic

:py:mod:`openmm_interface` contains the ``get_metatensor_force`` function,
which can be used to load a :py:class:`MetatensorAtomisticModel` into an
``openmm.Force`` object able to calculate forces on an ``openmm.System``.

.. autofunction:: metatensor.torch.atomistic.openmm_force.get_metatensor_force

In order to run simulations with ``metatensor.torch.atomistic`` and ``OpenMM``,
we recommend installing ``OpenMM`` from conda, using
``conda install -c conda-forge openmm-torch nnpops``. Subsequently,
metatensor can be installed with ``pip install metatensor[torch]``, and a minimal
script demonstrating how to run a simple simulation is illustrated below:

.. code-block:: python

    import openmm
    from metatensor.torch.atomistic.openmm_interface import get_metatensor_force

    # load an input geometry file
    topology = openmm.app.PDBFile('input.pdb').getTopology()
    system = openmm.System()
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)
    
    # get the force object from an exported model saved as 'model.pt'
    force = get_metatensor_force(system, topology, 'model.pt')
    system.addForce(force)

    integrator = openmm.VerletIntegrator(0.001)
    platform = openmm.Platform.getPlatformByName('CUDA')
    simulation = openmm.app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(openmm.app.PDBFile('input.pdb').getPositions())
    simulation.reporters.append(openmm.app.PDBReporter('output.pdb', 100))

    simulation.step(1000)
