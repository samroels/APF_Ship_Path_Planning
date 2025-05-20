Artificial Potential Field for Autonomous Ship Path Planning

This project implements a basic Artificial Potential Field (APF) algorithm for autonomous ship path planning in inland waterways, specifically simulating the Western Scheldt.

RUNNING THE PROGRAM

Make sure the requirements are installed by running "pip install -r requirements.txt"

To run the simulation use the command "phython ppo.py" in the terminal.


Simulation parameters can be chosen/adjusted in continuous.py

Key parameters:

k_att: The attraction strength towards the goal.

k_rep: The repulsive strength from obstacles.

inf_rad: The influence radius of repulsive forces.


FILES

- APF_Path_Planner.py
	Implements the Artificial Potential Field (APF) method for path generation, includes methods for attractive and repulsive 	potential calculations and path generation based on gradient descent
- continuous_env.py
	Main simulation environment, implements 2D ship navigation environment, defines ship dynamics, rendering and integrates 	the APF-generated path and potential field
- continuous_tools.py
	Utility functions used by the environment
- ppo.py
	Contains Proximal Policy Optimization (PPO) agent training and evaluation logic, visualizes trained agent's path but is 	commented out for now so only the generated path is shown
- env_Sche_250cm_no_scale.csv
	Environment polygon data representing the waterway walls used as APF repulsive obstacles
- env_Sche_no_scale.csv
	Environment polygon data for visualization of the broader area (the Western Scheldt)
- trajectory_points_no_scale.csv
	Original PPO-generated goal used as target in APF method
- requirements.txt
	Python package dependencies to install before running the program

TEST RESULTS of the project can be found in the folder
"\Portfolio_Bachelorthesis_Roels_Sam\Presentations\Weekly_meeting_presentations"

FUTURE PROPOSALS
The possible future improvements can be found in the conclusion of the thesis at
"
Acknowledgments
Inspired by research on APF in autonomous navigation.

Developed as part of a Bachelor's thesis at the University of Antwerp.