# 3D-PTV method for microplastic column experiments
This repository calibrates 2 high-definitions cameras based on a 2D checkerboard calibration object and allows for subsequent 3D particle velocimetry tracking (3D-PTV) in particle column experiments. 

## Requirements
All needed Python packages are specified in the *requirements.txt* file 

## Set-Up 
Source codes are provided in the [source](https://github.com/valeriederijk/3D-PTV-microplastic/tree/master/scripts) folder. Camera Calibration v1 and v2 differ in their selection of alphabetic frames or random frames. The following order should be followed when using the scripts: 
- *camera_sync_video.py* synchs the two input videos
- *definingcroppingparameters.py* defines the cropping parameters in order to isolate the column
- *extract_and_cropframes.py* extracts all frames and crops them to the parameters that are defined in defingingcroppingparameters.py
- *manual_particle_detection_improvement.py* allows you to manually adjust particle tracking thresholds that are used in the 3D tracking process
- *3d_tracking_final* identifies particles, appends them to trajectories and creates visual output for the particle tracking process
- *particleparameters.py* and *theoreticalapproximations.py* compute parameters needed for theoretical approximations.

## License 
Distributed under the MIT License. See *LICENSE.txt* for more information.

## Contact
Developed by Valerie de Rijk (v.derijk@uu.nl)
