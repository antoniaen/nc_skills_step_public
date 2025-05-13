"""Main analysis with optimal bandwidth according to CCT (2014).

The analysis is with the partially treated individuals in Vietnam.

"""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab


@pytask.mark.depends_on(
    {
        "reg_functions": SRC / "analysis" / "analysis_RDD.py",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "latex_tables": SRC / "final" / "latex_tables_with_regression_results.py",
        "global_info": SRC / "global_info.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        "optimal_bandwidth": BLD / "python" / "data" / "optimal_bandwidth_CCT.pkl",
    },
)
@pytask.mark.produces(
    {
        "with_estimates": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "results_optimal_bandwidth_CCT.tex",
        "without_estimates": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "results_optimal_bandwidth_CCT_bandwidths_only.tex",
    },
)
def task_analysis_with_optimal_bandwidth_CCT(depends_on, produces):
    """Latex tabular results with optimal bandwidth and partially treated."""
    y_vars = (
        ["years_educ"]
        + gl.groups_of_dependent_variables["ncogn_skills"]
        + gl.groups_of_dependent_variables["preferences_binary"]
    )
    results_dict = {key: None for key in y_vars}
    dep_var_names = {key: gl.nice_variable_names[key] for key in y_vars}

    data = pd.read_pickle(depends_on["data"])
    h_df = pd.read_pickle(depends_on["optimal_bandwidth"])

    for y_var in y_vars:
        reg_data = sel.select_sample_for_analysis(
            data=data,
            y_vars=[y_var],
            n_years=int(h_df.loc[y_var, "h (left)"]),
            reform_list=gl.reforms_final,
        )

        results_dict[y_var] = [
            reg.linear_flexible_trends(
                data=reg_data,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            ),
        ]

    tab.create_table_with_optimal_bandwidth(
        file=str(produces["with_estimates"]).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=["Treated", "SE", "Optimal bandwidth", "N"],
        dep_var_names=dep_var_names,
        h_df=h_df,
    )
    tab.create_table_with_optimal_bandwidth(
        file=str(produces["without_estimates"]).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=["Optimal bandwidth", "N"],
        dep_var_names=dep_var_names,
        h_df=h_df,
        with_estimates=False,
    )
