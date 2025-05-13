# The impact of mandatory education on non-cognitive skills: Evidence from low- and middle-income countries

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/antoniaen/nc_skills_step_public/main.svg)](https://results.pre-commit.ci/latest/github/antoniaen/nc_skills_step_public/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Environment

To get started, navigate to the parent folder of the project and create the environment
called `nc_skills_step_public` with:

```console
$ conda env create -f environment.yml
```

or

```console
$ mamba env create -f environment.yml
```

Then activate the environment via:

```console
$ conda activate nc_skills_step_public
```

## Data

You have to save the original data in the folder `src/nc_skills_step_public/data`. You need the
following documents

- STEP Bolivia_working.dta
- STEP Colombia_working.dta
- STEP Ghana_working.dta
- STEP Vietnam_working.dta

which are public use data files from the STEP Skills Measurement Household Surveys that
can be requested
[here](https://microdata.worldbank.org/index.php/catalog/step/?page=1&ps=15&repo=step).

## Building the project

To build the project run

```console
$ pytask
```

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
