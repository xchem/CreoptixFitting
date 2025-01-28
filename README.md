# CreoptixFitting

Original documentation in [gciscripts/README.md](gciscripts/README.md)

## Usage at DLS

### Submitting jobs

From a DLS linux machine terminal (or NoMachine / ssh):

#### 1. Load the python environment:

```
source /dls/science/groups/i04-1/software/max/load_py310.sh
```

#### 2. Navigate/create a working directory

e.g.

```
mkdir creoptix_test
cd creoptix_test
pwd
```

#### 3. Place your input files inside:

- *trace file*, a text file exported from Creoptix containing response time-series for all channels
- *schema file*, a text file containing the copied and pasted table describing the channels from Creoptix

To use the example data:

```
cp -v $CREOPTIX/example/input/sample_*_3.txt .
```

#### 4. Copy and edit your config file:

```
cp -v $CREOPTIX/config.json .
nano config.json
```

#### 5. See the CLI help:

```
python -m gcifit.fit --help
```

#### 6. Submit the fitting

```
python -m gcifit.fit sample_traces_3.txt sample_schema_3.txt config.json example_output
```

#### 7. Track running jobs:

```
rq
```

### Viewing outputs in a notebook

From a DLS linux machine terminal (or NoMachine / ssh):

#### 1. Load the python environment:

```
source /dls/science/groups/i04-1/software/max/load_py310.sh
```

#### 2. Change to the CreoptixFitting directory:

```
cd $CREOPTIX
```

#### 3. Launch a jupyter notebook

```
jupyter lab
```

#### 4. Open the address shown in the terminal in a browser

#### 5. Try out the example notebook: `show_outputs.ipynb`
