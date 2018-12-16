# Metrics for Graph Comparison
### Peter Wills, Ph.D.

This repository contains the code used in the paper [Metrics for Graph Comparison: a Practitioner's Guide](http://www.pwills.com/404). Some of the work is done in Python scripts, but most is done directly in notebooks.

## Usage

This code is packaged via `pipenv`, so if you don't have it, do `pip install pipenv`. (If you don't have Python or pip, I recommend [Miniconda 3](https://conda.io/miniconda.html)).

The scripts and notebooks should be run inside the virtual environment specified in the `Pipfile`. To run the data scripts, do (for example)

	pipenv run python data_scripts/abide.py
	
To start a jupyter notebook server, do

	pipenv run jupyter notebook

## Data Generation & Visualization

There are four distinct sections which are supported by data figures. 

### Evaluation of Distances on Random Graph Ensembles

Section 4 generates samples from random graph ensembles, and compares them, in order to elucidate how graph structure informs the effectiveness of various distances. 

The overall workflow for generating the data is that (1) we sample random graphs and compare them in parallel, and then (2) we generate boxplots of the resulting distances in order to show their statistical properties visually. The scripts that generate the samples are in `data_scripts`. The results of running these scripts is in the directory `pickled_data`, but this can be re-generated by running the appropriate scripts. The plotting is done in the notebook `random_graph_ensembles.ipynb`. 

Figure 4.10 shows empirical spectral densities for a variety of random graph ensembles; these are generated and plotted in `spectral_density.ipynb`.

### Primary School Social Graph

We examine two empirical datasets; the first is a social graph from a French primary school. The data is plotted in `primary_school.ipynb`. The data is in `data`, but you can download it yourself via a `curl` shown in the notebook.

### ABIDE Connectome Data

We also use data from the Autism Brain Image Data Exchange, or ABIDE. This compares connectome data of brains with and without autism spectrum disorder. The distances between connectomes in the two groups are generated via `data_scripts/abide.py`. The distance-comparison boxplots are generated in `abide_figs.ipynb`; the correlation heatmap is generated in `abide_structures.ipynb`.


## Credits
 
Author: Peter Wills (peter@pwills.com)
 
## License
 
The MIT License (MIT)

Copyright (c) 2018 Peter Wills

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.