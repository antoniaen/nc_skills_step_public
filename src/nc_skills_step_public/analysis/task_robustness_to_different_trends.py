"""LaTeX tables with results with other cohort/age trends.

First, we study more flexible trends. Then we study trends that are inflexible at the
cutoff.

"""


import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

for group in ("ncogn_skills", "preferences_binary"):
    kwargs = {
        "group": group,
        "produces": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / f"results_with_more_flexible_trends_{group}.tex",
    }

    @pytask.mark.depends_on(
        {
            "scripts": [
                "select_sample_for_analysis.py",
                "analysis_RDD.py",
            ],
            "latex_table": SRC / "final" / "latex_tables_with_regression_results.py",
            "global_info": SRC / "global_info.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_robustness_more_flexible_trends(
        depends_on,
        group,
        produces,
    ):
        """Latex tabular with results from more flexible cohort/age trends."""
        dep_vars = gl.groups_of_dependent_variables[group]
        results_dict = {key: None for key in dep_vars}
        dep_var_names = {key: gl.nice_variable_names[key] for key in dep_vars}

        data = pd.read_pickle(depends_on["data"])

        # 3 years
        reg_data_3y = sel.select_sample_for_analysis(
            data=data,
            y_vars=dep_vars,
            n_years=3,
            reform_list=gl.reforms_final,
        )
        # 5 years
        reg_data_5y = sel.select_sample_for_analysis(
            data=data,
            y_vars=dep_vars,
            n_years=5,
            reform_list=gl.reforms_final,
        )
        # 10 years
        reg_data_10y = sel.select_sample_for_analysis(
            data=data,
            y_vars=dep_vars,
            n_years=10,
            reform_list=gl.reforms_final,
        )

        for y_var in dep_vars:
            results_3y_2 = reg.quadratic_flexible_trends(
                data=reg_data_3y,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            )
            results_3y_3 = reg.cubic_flexible_trends(
                data=reg_data_3y,
                y_var=y_var,
            )
            results_5y_2 = reg.quadratic_flexible_trends(
                data=reg_data_5y,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            )
            results_5y_3 = reg.cubic_flexible_trends(
                data=reg_data_5y,
                y_var=y_var,
            )
            results_5y_4 = reg.quartic_flexible_trends(
                data=reg_data_5y,
                y_var=y_var,
            )
            results_10y_2 = reg.quadratic_flexible_trends(
                data=reg_data_10y,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            )
            results_10y_3 = reg.cubic_flexible_trends(
                data=reg_data_10y,
                y_var=y_var,
            )
            results_10y_4 = reg.quartic_flexible_trends(
                data=reg_data_10y,
                y_var=y_var,
            )

            results_dict[y_var] = [
                results_10y_2,
                results_10y_3,
                results_10y_4,
                results_3y_2,
                results_3y_3,
                results_5y_2,
                results_5y_3,
                results_5y_4,
            ]

        column_headers = [
            "quad.",
            "cub.",
            "quar.",
            "quad.",
            "cub.",
            "quad.",
            "cub.",
            "quar.",
        ]

        tab.create_tabular_tex_code_with_reg_results(
            file=str(produces).replace(".tex", ""),
            results=results_dict,
            regressor="treated",
            column_headers=column_headers,
            dep_var_names=dep_var_names,
            version=3,
            gen_pdf=False,
        )


for group in ("ncogn_skills", "preferences_binary"):
    kwargs = {
        "group": group,
        "produces": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / f"results_with_inflexible_trends_at_cutoff_{group}.tex",
    }

    @pytask.mark.depends_on(
        {
            "scripts": [
                "select_sample_for_analysis.py",
                "analysis_RDD.py",
            ],
            "latex_table": SRC / "final" / "latex_tables_with_regression_results.py",
            "global_info": SRC / "global_info.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_robustness_inflexible_trends(
        depends_on,
        group,
        produces,
    ):
        """Latex tabular results from trends that are inflexible at the cutoff."""
        dep_vars = gl.groups_of_dependent_variables[group]
        results_dict = {key: None for key in dep_vars}
        dep_var_names = {key: gl.nice_variable_names[key] for key in dep_vars}

        data = pd.read_pickle(depends_on["data"])

        # 3 years
        reg_data_3y = sel.select_sample_for_analysis(
            data=data,
            y_vars=dep_vars,
            n_years=3,
            reform_list=gl.reforms_final,
        )
        # 5 years
        reg_data_5y = sel.select_sample_for_analysis(
            data=data,
            y_vars=dep_vars,
            n_years=5,
            reform_list=gl.reforms_final,
        )
        # 10 years
        reg_data_10y = sel.select_sample_for_analysis(
            data=data,
            y_vars=dep_vars,
            n_years=10,
            reform_list=gl.reforms_final,
        )

        for y_var in dep_vars:
            results_3y = reg.linear_inflexible_trends(
                data=reg_data_3y,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            )
            results_5y = reg.linear_inflexible_trends(
                data=reg_data_5y,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            )
            results_10y = reg.linear_inflexible_trends(
                data=reg_data_10y,
                y_var=y_var,
                reform_type_dummy=False,
                partially_treated=True,
            )

            results_dict[y_var] = [results_5y, results_3y, results_10y]

        column_headers = ["5 years", "3 years", "10 years"]

        tab.create_tabular_tex_code_with_reg_results(
            file=str(produces).replace(".tex", ""),
            results=results_dict,
            regressor="treated",
            column_headers=column_headers,
            dep_var_names=dep_var_names,
            version=None,
            gen_pdf=False,
        )
