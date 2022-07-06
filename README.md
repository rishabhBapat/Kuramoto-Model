# Kuramoto Model

A python implementation of the kuramoto model.

## Installation

```
pip install kuramoto_model
```
## Usage

Import the model,
```python
from kuramoto_model.kuramoto_model import Kuramoto
```
Initialise the model with the following,
1. Number of neurons `n`
2. Coupling constant `k`
3. Timeseries `timeseries`: the timepoints to log results at
4. Intrinsic Frequencies `omega_n`: defaults to n random values from a normal distribution
5. Initial Phases `theta_n`: defaults to n random values between 0 and 2pi
6. Adjacency Matrix `adjacency_nxn`: defaults to all to all coupling (without self-coupling)
```python
n = 100
k = 0.8
ts = np.linspace(0, 100, 1000)
model = Kuramoto(n, k, ts)

```
Find the phase, coherence and mean frequency timeseries,
```python
phases = model.phase_timeseries()
coherences = model.coherence_timeseries()
mean_freq = model.mean_frequency_timeseries()
``` 
