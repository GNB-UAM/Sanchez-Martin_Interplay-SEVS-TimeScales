# Sanchez-Martin_Interplay-SEVS-TimeScales
Code and datasets generated in Sanchez-Martin et al. publication titled "Interplay among Neural Synchronization, Excitability,Variability, and Sequentiality across Time Scales in Rhythmic Bursting". 
This code includes all data analysis and figure plotting.

If you use this code please cite: Sanchez-Martin, P., Garrido-Peña, A., Elices, I., Garcia-Saura, C., Levi, R., Rodriguez, F.B., Varona, P. (2026). Interplay among Neural Synchronization, Excitability,Variability, and Sequentiality across Time Scales in Rhythmic Bursting [Manuscript submitted for publication].

## How to use
Load and activate conda environment from environment.yml file with the following commands:

	conda env create -f environment.yml
	conda activate interplay_metrics

### Figures
To recreate each figure run the corresponding script in the Figures folder like:

	python3 cycle_by_cycle.py
	
All scripts process the analyzed_data.pkl file located in the main folder.

