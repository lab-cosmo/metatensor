import os
from importlib import reload

import pytest

import metatensor


def set_initial_state(state):
    metatensor.operations._checks._CHECKS_ENABLED = state


@pytest.mark.parametrize("inital_state", [True, False])
def test_checks_enabled(inital_state):
    """Test checks_enabled() function."""
    set_initial_state(inital_state)
    assert metatensor.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_enable_checks(inital_state):
    """Test unsafe_enable_checks() function."""
    set_initial_state(inital_state)
    metatensor.unsafe_enable_checks()
    assert metatensor.checks_enabled() is True


def test_enable_repr():
    assert metatensor.unsafe_enable_checks().__repr__() == "checks enabled"


def test_disable_repr():
    assert metatensor.unsafe_disable_checks().__repr__() == "checks disabled"


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_disable_checks(inital_state):
    """Test unsafe_disable_checks() function"""
    set_initial_state(inital_state)
    metatensor.unsafe_disable_checks()
    assert metatensor.checks_enabled() is False


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_enable_checks_context_manager(inital_state):
    """Test unsafe_enable_checks() context manager"""
    set_initial_state(inital_state)
    with metatensor.unsafe_enable_checks():
        assert metatensor.checks_enabled() is True
    assert metatensor.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_unsafe_disable_checks_context_manager(inital_state):
    """Test unsafe_disable_checks() context manager"""
    set_initial_state(inital_state)
    with metatensor.unsafe_disable_checks():
        assert metatensor.checks_enabled() is False
    assert metatensor.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_nested_context_managers_enable_disable(inital_state):
    """Test nested enable-disable context managers"""
    set_initial_state(inital_state)
    with metatensor.unsafe_enable_checks():
        assert metatensor.checks_enabled() is True
        with metatensor.unsafe_disable_checks():
            assert metatensor.checks_enabled() is False
        assert metatensor.checks_enabled() is True
    assert metatensor.checks_enabled() is inital_state


@pytest.mark.parametrize("inital_state", [True, False])
def test_nested_context_managers_disable_enable(inital_state):
    """Test nested enable-disable context managers"""
    set_initial_state(inital_state)
    with metatensor.unsafe_disable_checks():
        assert metatensor.checks_enabled() is False
        with metatensor.unsafe_enable_checks():
            assert metatensor.checks_enabled() is True
        assert metatensor.checks_enabled() is False
    assert metatensor.checks_enabled() is inital_state


@pytest.mark.parametrize("original_state", [True, False])
def test_environment_variable(original_state):
    """Test environment variable METATENSOR_unsafe_disable_checks"""
    os.environ["METATENSOR_UNSAFE_DISABLE_CHECKS"] = str(not original_state)

    reload(metatensor.operations._checks)
    assert metatensor.checks_enabled() == original_state
