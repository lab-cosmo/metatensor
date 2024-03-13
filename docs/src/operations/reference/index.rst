.. _python-api-operations:

API reference
=============

.. note::

  This is the documentation for ``metatensor-operations`` version
  |metatensor-operations-version|. For other versions, look in the following
  pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-version:: 0.2.1
        :tag-prefix: metatensor-operations-v
        :url-suffix: operations/reference/index.html

    .. grid-item-version:: 0.2.0
        :tag-prefix: metatensor-operations-v
        :url-suffix: operations/reference/index.html

    .. grid-item-version:: 0.1.0
        :tag-prefix: metatensor-operations-v
        :url-suffix: reference/operations/index.html


All operations are automatically re-exported from
``metatensor.operations.<xxx>`` as ``metatensor.<xxx>`` when using the Python
backend; and as ``metatensor.torch.<xxx>``` when using the :ref:`TorchScript
backend <operations-and-torch>`.

.. toctree::
    :maxdepth: 2

    creation/index
    linear_algebra/index
    logic/index
    manipulation/index
    math/index
    set/index
    checks
