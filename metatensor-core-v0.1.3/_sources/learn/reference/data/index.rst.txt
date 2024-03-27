Data utilites
=============

* :ref:`learn-data-dataset`
* :ref:`learn-data-dataloader`
* :ref:`learn-data-collate`

.. _learn-data-dataset:

Dataset
-------

.. autoclass:: metatensor.learn.Dataset
   :members:
   :special-members: __getitem__

.. autoclass:: metatensor.learn.IndexedDataset
   :members:
   :special-members: __getitem__

.. _learn-data-dataloader:

Dataloader
----------

.. autoclass:: metatensor.learn.DataLoader
   :members:

.. _learn-data-collate:

Collating data
--------------

.. autofunction:: metatensor.learn.data.group

.. autofunction:: metatensor.learn.data.group_and_join
