# CreoptixFitting

Original documentation in [gciscripts/README.md](gciscripts/README.md)

## Usage at DLS

### Submitting jobs

From a DLS linux machine terminal (or NoMachine/ssh):

#### 1. Connect to the SLURM cluster "wilson"

```
ssh wilson
```

#### 2. Load the python environment:

```
source /dls/science/groups/i04-1/software/max/load_py310.sh
```

#### 3. Navigate/create a working directory

e.g.

```
mkdir creoptix_test
cd creoptix_test
pwd
```

#### 4. Place your input files inside:

- *trace file*, a text file exported from Creoptix containing response time-series for all channels
- *schema file*, a text file containing the copied and pasted table describing the channels from Creoptix

To use the example data:

```
cp -v $CREOPTIX/example/input/sample_*_3.txt .
```

#### 5. Copy and edit your config file:

```
cp -v $CREOPTIX/config.json .
nano config.json
```

#### 6. See the CLI help:

```
python -m gcifit.fit --help
```

#### 7. Submit the fitting

```
python -m gcifit.fit sample_traces_3.txt sample_schema_3.txt config.json example_output
```

#### 8. Track running jobs:

```
rq
```

### Viewing outputs in a notebook

From a DLS linux machine terminal (or NoMachine):

#### 1. Load the python environment:

```
source /dls/science/groups/i04-1/software/max/load_py310.sh
```

#### 2. Change to a working directory:

e.g.

```
cd creoptix_test
```

#### 3. Copy over the example notebook file

```
cp $CREOPTIX/show_outputs.ipynb .
```

#### 4. Launch a jupyter notebook

```
jupyter lab
```

#### 5. Open the address shown in the terminal in a browser (if it doesn't open automatically)

#### 6. Try out the example notebook: `show_outputs.ipynb`
