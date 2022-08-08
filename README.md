1. Generate 1d mesh with data_mesh.ipynb (change `pref_folder`, where you want to save mesh data)

2. Generate 1d density profile with data_preprocess.ipynb (change `pref_folder`, where you want to save mesh data, explore `bw_method`, which smoothes the data)

3. Train 1d neural network (change `PATH`, where you save data)
```
python train_1d.py
```
4. Visualize on test data data_validate.ipynb (change `PATH`, where you save data)

5. Design optimization with data_design.ipynb


