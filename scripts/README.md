# Scripts
- [semantic_query_cli](semantic_query_cli.py): CLI to build a RayFronts map from MCAP/live ROS2/dataset streams and return object position from top-k salient semantic voxels.
- [verify_query_overlay](verify_query_overlay.py): Projects `semantic_query_cli` output (`estimated_xyz` + salient voxels) back into MCAP RGB frames, scores visibility/salience across frames, and saves top-ranked overlay images (with frame skipping support).
- [semseg_eval](semseg_eval.py): Script to perform open-vocabulary zero shot semantic segmentation evaluation. Has an option to load external point cloud predictions for evaluation giving the flexibility to evaluate approaches not ported to this repo.
- [srchvol_eval](srchvol_eval.py): Script to perform the search volume online evaluation for the mapping baselines evaluated in RayFronts.
- [summarize_srchvol_eval](summarize_srchvol_eval.py): Rough script used to post process srchvol_eval results and compute AUC metrics and visualizations.
- [fit_feat_compressor](fit_feat_compressor.py): Used to fit a feature compressor to some data distribution. (e.g computing a PCA basis for some encoder on some data for better visualization or better compression.)

Deprecated
- [ros2npy](ros2npy.py): Used to convert ROS1 ZedX bags to numpy arrays to use with the rosnpy dataloader.
