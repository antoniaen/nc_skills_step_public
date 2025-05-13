"""Results figure for the paper."""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import plots as pl


@pytask.mark.depends_on(
    {
        "scripts": ["plots.py"],
        "global_info": SRC / "global_info.py",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "reg_functions": SRC / "analysis" / "analysis_RDD.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "results_figure_two_xaxis.png")
def task_results_plot_with_two_xaxis(depends_on, produces):
    """Create figure with estimated coefficients and two x-axis.

    One x-axis is for standardized data (personality and behaviors) and the other is for
    binary data (risk and patience).

    """
    data = pd.read_pickle(depends_on["data"])

    results_list = []

    y_vars = (
        gl.groups_of_dependent_variables["ncogn_skills"]
        + gl.groups_of_dependent_variables["preferences_binary"]
    )
    data_subset_ncogn = sel.select_sample_for_analysis(
        data=data,
        y_vars=gl.groups_of_dependent_variables["ncogn_skills"],
        n_years=5,
        reform_list=gl.reforms_final,
    )
    data_subset_pref = sel.select_sample_for_analysis(
        data=data,
        y_vars=gl.groups_of_dependent_variables["preferences_binary"],
        n_years=5,
        reform_list=gl.reforms_final,
    )

    for y_var in y_vars:
        if y_var in gl.groups_of_dependent_variables["ncogn_skills"]:
            data_subset = data_subset_ncogn
        elif y_var in gl.groups_of_dependent_variables["preferences_binary"]:
            data_subset = data_subset_pref

        results = reg.linear_flexible_trends(
            data=data_subset,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_list.append(results)

    fig = pl.coeff_plot_two_xaxis(
        results_list=results_list,
        nice_variable_names=gl.nice_variable_names,
        group_mapping=gl.nice_variable_names_to_broad_groups_mapping,
    )
    fig.write_image(produces, scale=2)
