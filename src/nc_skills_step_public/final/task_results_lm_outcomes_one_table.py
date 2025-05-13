"""LaTeX tables with multiple regression results in one table.

Partially treated individuals are included.

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
        "scripts": ["latex_tables_with_regression_results.py"],
        "reg_functions": SRC / "analysis" / "analysis_RDD.py",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "global_info": SRC / "global_info.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(
    BLD
    / "python"
    / "tables"
    / "with_partially_treated"
    / "common_trend"
    / "results_all_lm_outcomes_one_table.tex",
)
def task_results_lm_outcomes_one_table(depends_on, produces):
    """Latex tabular with regression results for labor market outcomes."""
    dep_vars = ["ln_earnings_h_usd", "emp", "wage_worker"]
    results_dict = {key: None for key in dep_vars}
    dep_var_names = {key: gl.nice_variable_names[key] for key in dep_vars}

    data = pd.read_pickle(depends_on["data"])

    for y_var in dep_vars:
        # 3 years
        reg_data_3y = sel.select_sample_for_analysis(
            data=data,
            y_vars=y_var,
            n_years=3,
            reform_list=gl.reforms_final,
        )
        # 5 years
        reg_data_5y = sel.select_sample_for_analysis(
            data=data,
            y_vars=y_var,
            n_years=5,
            reform_list=gl.reforms_final,
        )
        # 10 years
        reg_data_10y = sel.select_sample_for_analysis(
            data=data,
            y_vars=y_var,
            n_years=10,
            reform_list=gl.reforms_final,
        )

        results_3lin = reg.linear_flexible_trends(
            data=reg_data_3y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_3quad = reg.quadratic_flexible_trends(
            data=reg_data_3y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_5lin = reg.linear_flexible_trends(
            data=reg_data_5y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_5quad = reg.quadratic_flexible_trends(
            data=reg_data_5y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_10lin = reg.linear_flexible_trends(
            data=reg_data_10y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_10quad = reg.quadratic_flexible_trends(
            data=reg_data_10y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_10cub = reg.cubic_flexible_trends(
            data=reg_data_10y,
            y_var=y_var,
        )

        results_dict[y_var] = [
            results_5lin,
            results_5quad,
            results_3lin,
            results_3quad,
            results_10lin,
            results_10quad,
            results_10cub,
        ]

    column_headers = [
        "lin",
        "quad",
        "lin",
        "quad",
        "lin",
        "quad",
        "cub",
    ]

    tab.latex_tabular_multiple_outcomes(
        file=str(produces).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=column_headers,
        dep_var_names=dep_var_names,
        version=4,
    )
