"""Convert EgoHumanoid human-motion HDF5 episodes into a VLA-finetune-ready
LeRobot v2.1 dataset for the GR00T whole-body-control workflow.

Pipeline (staged because no single env has every dependency):

  stage1_retarget   (conda env `gmr`)            HDF5 -> G1 motion .pkl
  stage2_video      (.venv_sim, pyzed)           .svo2 -> ego_view frames .npz
  stage3_tokens     (.venv_sim, mujoco+ort)      .pkl -> action.motion_token .npz
  stage4_assemble   (.venv_data_collection)      .pkl + frames + tokens -> LeRobot

See ``run.py`` for the orchestrator that invokes each stage in its env.
"""
