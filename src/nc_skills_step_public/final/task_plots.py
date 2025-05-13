"""Plots."""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_other_regressions as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import plots as pl


@pytask.mark.depends_on(
    {
        "scripts": ["plots.py"],
        "global_info": SRC / "global_info.py",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "RDD_plot.png")
def task_plot_years_of_education(depends_on, produces):
    """Plot years of education around the reforms."""
    data = pd.read_pickle(depends_on["data"])

    data_subset = sel.select_sample_for_analysis(
        data=data,
        y_vars=["years_educ"],
        n_years=5,
        reform_list=gl.reforms_final,
    )

    fig = pl.plot_years_of_education(
        data=data_subset,
        staggered=["Vietnam1991"],
        not_staggered=["Ghana1961", "Ghana1987", "Colombia1991", "Bolivia1994"],
    )
    fig.write_image(produces, width=800, height=500, scale=2)


for y_var in gl.dependent_variables:
    kwargs = {
        "y_var": y_var,
        "produces": {
            "years": BLD / "python" / "figures" / f"RDD_plot_{y_var}.png",
            "months": BLD / "python" / "figures" / f"RDD_plot_{y_var}_months.png",
        },
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["plots.py"],
            "global_info": SRC / "global_info.py",
            "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
            "reg": SRC / "analysis" / "analysis_RDD.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=y_var, kwargs=kwargs)
    def task_plot(depends_on, y_var, produces):
        """Plot outcome around the cutoffs."""
        data = pd.read_pickle(depends_on["data"])

        # Birth years on x-axis.
        data_subset = sel.select_sample_for_analysis(
            data=data,
            y_vars=[y_var],
            n_years=5,
            reform_list=gl.reforms_final,
        )

        results = reg.fit_for_RDD_plot(
            data=data_subset,
            y_var=y_var,
            partially_treated=True,
        )

        fig = pl.plot_outcomes(
            data=data_subset,
            outcome=y_var,
            reg_results=results,
            nice_variable_names=gl.nice_variable_names,
        )
        fig.write_image(produces["years"], width=1000, height=400, scale=2)

        # Birth months on x-axis.
        data_subset = sel.select_sample_for_analysis_months_based(
            data=data,
            y_vars=[y_var],
            n_months=60,
            reform_list=gl.reforms_final,
        )

        results = reg.fit_for_RDD_plot(
            data=data_subset,
            y_var=y_var,
            months=True,
            partially_treated=True,
        )

        fig = pl.plot_outcomes(
            data=data_subset,
            outcome=y_var,
            reg_results=results,
            nice_variable_names=gl.nice_variable_names,
            months=True,
        )
        fig.write_image(produces["months"], width=800, height=500, scale=2)


@pytask.mark.depends_on(
    {
        "scripts": ["plots.py"],
        "global_info": SRC / "global_info.py",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(
    {
        "occupation": BLD / "python" / "figures" / "occupations_isco.png",
        "occtype_step": BLD / "python" / "figures" / "occupations_step.png",
    },
)
def task_plot_occupations(depends_on, produces):
    """Histograms with occupations."""
    data = pd.read_pickle(depends_on["data"])

    data_subset = sel.select_sample_for_analysis(
        data=data,
        y_vars=["occupation"],
        n_years=5,
        reform_list=gl.reforms_final,
    )

    for occ_type in ["occupation", "occtype_step"]:
        fig = pl.hist_occupations(data=data_subset, occ_type=occ_type)
        fig.write_image(produces[occ_type], width=800, height=500, scale=2)
