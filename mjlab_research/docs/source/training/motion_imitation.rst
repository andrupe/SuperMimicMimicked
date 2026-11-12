.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

WandB registry setup
--------------------

mjlab uses `Weights & Biases <https://wandb.ai/>`_ to store and load
reference motions. Before preprocessing any motions, create a WandB registry
by following the
`BeyondMimic instructions <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
(only the registry creation step; skip the ``csv_to_npz.py`` command shown
there).

Motion preprocessing
--------------------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

Convert a CSV to the NPZ format mjlab expects:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

The script resamples the motion to the output frame rate and uploads the
resulting NPZ to your WandB registry.

.. warning::

   You **must** use mjlab's converter (``mjlab.scripts.csv_to_npz``). The NPZ
   stores the root link's world pose and twist and the joint states, with
   ``joint_names`` in column order; loading raises unless those names equal the
   robot's joint names, so NPZ files from other frameworks do not load.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
