import ase.units

from metatensor.torch.atomistic import unit_conversion_factor


def test_conversion_length():
    length_angstrom = 1.0
    length_nm = unit_conversion_factor("length", "angstrom", "nm") * length_angstrom
    assert length_nm == 0.1


def test_conversion_energy():
    energy_ev = 1.0
    energy_mev = unit_conversion_factor("energy", "ev", "mev") * energy_ev
    assert energy_mev == 1000.0


def test_units():
    def length_conversion(unit):
        return unit_conversion_factor("length", "angstrom", unit)

    assert length_conversion("bohr") == ase.units.Ang / ase.units.Bohr
    assert length_conversion("nm") == ase.units.Ang / ase.units.nm
    assert length_conversion("nanometer") == ase.units.Ang / ase.units.nm

    def energy_conversion(unit):
        return unit_conversion_factor("energy", "ev", unit)

    assert energy_conversion("Hartree") == ase.units.eV / ase.units.Hartree
    kcal_mol = ase.units.kcal / ase.units.mol
    assert energy_conversion("kcal/mol") == ase.units.eV / kcal_mol
    kJ_mol = ase.units.kJ / ase.units.mol
    assert energy_conversion("kJ/mol") == ase.units.eV / kJ_mol
