Implementation of models from Kefi et. al., "Early Warning Signals of Ecological Transitions: Methods for Spatial Patterns" (2014)
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0092097


kefi_et_al_models.py contains definitions of the three models and functions for simulating them

kefi_generate_data.py is a wrapper which runs the specified model and outputs data locally to .pkl files

kefi_model_animate.py produces animations to visualize simulation results and saves local .mp4 files


As of the time I am writing this, only Model #1 is producing a clear phase transition as hoped. I have left most model parameters as specified in Appendix S2 of the Kefi paper, but in some cases values were not provided and in others I have tweaked them. 
