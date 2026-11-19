# Tuning the simulator

- Edit [tuning.py](tuning.py) for waveform parameter ranges and morphology rules.
- Edit [validators.py](validators.py) for the feature acceptance criteria.
- Edit [plotting.py](plotting.py) for image rendering.
- Edit [simulator.py](simulator.py) for sampling and rejection behavior.

`BASE_SCAFFOLD_RANGES` controls the shared waveform. `CLASS_PARAM_RANGES` defines family-specific ranges. `_apply_morphology_rules` applies morphology constraints before validation.

After a change, generate a small synthetic dataset in a new location and inspect both the images and exported labels. Do not weaken validation merely to increase acceptance. Changes to morphology, rendering, or sampling define a new dataset and should use a new experiment directory.
