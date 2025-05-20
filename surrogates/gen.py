import numpy as np
for i in range(7,60):
	for j in np.linspace(0.0, 1.0, 100).tolist():
		print('sbatch surrogate_base_c_p.sbatch 42 0.5',j,i)
