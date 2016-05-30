https://www.cs.drexel.edu/~kon/introcompvis/assignments/project3/

# Generate Tracking Points
python track_points.py N_POINTS_TO_TRACK IMAGE_1 IMAGE_2 ... IMAGE_N

# Re-Running Tests
Each of the tests (1 to 4) has cooresponding bash scripts to re-run the SFM and TrackPoints.

Within each of the test folders, you will find a ```sfm.sh``` bash script, and a ```track_points.sh``` script.
The ```track_points.sh``` is used to kick off the process of tracking points.
The ```sfm.sh``` is used to execute the sfm.py wrapper with the existing data.

Each of these scripts is intended to be executed from the root of this project, and make use of relative paths.

To execute test 1, from the project root, execute the following:
```
$ ./test1/sfm.sh
```
