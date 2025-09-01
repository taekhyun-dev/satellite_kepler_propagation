# Satellite Kepler Propagation

This repository contains a collection of scripts for propagating satellite orbits using Keplerian elements.  Several utilities are provided for generating Walker Delta constellations and creating temporary TLE files.  Example data used by the scripts is stored in the `data/` directory.

## Requirements
See `requirements.txt` for a list of Python dependencies.

## Usage
Individual Python files demonstrate different propagation scenarios.  For example:

```bash
python kepler_propagator.py
```

will run a basic propagation example using PyAstronomy.

## Federated Learning Settings

Some simulations under `simserver/` support federated learning.  The following
environment variables adjust optional behaviours:

- `FL_FEDPROX_MU`: non-zero value enables FedProx proximal regularisation with
  the given coefficient.
- `FL_OFFLINE_CONTINUE`: when set to `1`, completed local trainings are
  re-enqueued so that nodes continue learning even while offline.