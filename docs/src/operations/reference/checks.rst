Checks
======


Metatensor performs *shape* as well as *consistency* checks for the metadata
during each operation.

There are plans to enable and disable the checks with the functions documented
here to bypass some of them, but they are not currently implemented.

.. The check status can be verified with
.. :py:func:`metatensor.checks_enabled()` and controlled by
.. :py:class:`metatensor.unsafe_enable_checks()` and
.. :py:class:`metatensor.unsafe_disable_checks()`.

.. .. warning::

..     Using metatensor without checks is unsafe, can lead to unwanted results,
..     and may produce cryptic error messages. Therefore, disabling checks is not
..     recommended and should only be used by advanced users! If you see strange
..     results after disabling checks, try running the code again with checks
..     enabled.


.. Checks can either be disabled temporarily via a compound statements or globally
.. and permanently by calling the specific function directly.

.. The checks can also controlled wit the environment variable
.. ``METATENSOR_UNSAFE_DISABLE_CHECKS``. To disable checks set

.. .. code-block:: bash

..     export METATENSOR_UNSAFE_DISABLE_CHECKS=1


.. Note that :py:class:`metatensor.unsafe_enable_checks()` or
.. :py:class:`metatensor.unsafe_disable_checks()` overwrite the definition of the
.. environment variable.

.. autofunction:: metatensor.checks_enabled

.. autofunction:: metatensor.unsafe_disable_checks

.. autofunction:: metatensor.unsafe_enable_checks
