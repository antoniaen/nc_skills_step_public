"""LaTeX tables with multiple regression results in one table.

The sample also contains <=23 years old individuals. Partially treated individuals are
included.

"""

import pandas as pd
import pylatex as pl
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

for group in gl.groups_of_dependent_variables:
    kwargs = {
        "group": group,
        "produces": {
            "tex": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "without_restricting_age"
            / f"results_with_partially_treated_{group}.tex",
            "pdf": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "without_restricting_age"
            / f"results_with_partially_treated_{group}.pdf",
            "tex_tabular": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "without_restricting_age"
            / f"results_with_partially_treated_{group}_tabular.tex",
        },
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["latex_tables_with_regression_results.py"],
            "reg_functions": SRC / "analysis" / "analysis_RDD.py",
            "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
            "global_info": SRC / "global_info.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_latex_results_wo_age_restriction(
        depends_on,
        group,
        produces,
    ):
        """Latex tabular with regression results."""
        dep_vars = gl.groups_of_dependent_variables[group]
        results_dict = {key: None for key in dep_vars}
        dep_var_names = {key: gl.nice_variable_names[key] for key in dep_vars}

        data = pd.read_pickle(depends_on["data"])

        # 3 years
        reg_data_3y = sel.select_sample_for_robustness_check_wo_age_restriction(
            data=data,
            y_vars=dep_vars,
            n_years=3,
            reform_list=gl.reforms_final,
        )
        # 5 years
        reg_data_5y = sel.select_sample_for_robustness_check_wo_age_restriction(
            data=data,
            y_vars=dep_vars,
            n_years=5,
            reform_list=gl.reforms_final,
        )
        # 10 years
        reg_data_10y = sel.select_sample_for_robustness_check_wo_age_restriction(
            data=data,
            y_vars=dep_vars,
            n_years=10,
            reform_list=gl.reforms_final,
        )

        for y_var in dep_vars:
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

        # A table showing the regressors instead of the dependent variables in the first column.
        if group == "years_educ":
            tab.create_tabular_tex_code_reg_results_show_regressors(
                file=str(produces["tex_tabular"]).replace(".tex", ""),
                results=results_dict,
                regressor="treated",
                column_headers=column_headers,
                reg_var_names={
                    "treated": "Treated",
                    "partially_treated": pl.NoEscape(
                        r"Treated $\times$ Partially treated",
                    ),
                },
                additional_regressor="partially_treated",
                version=4,
            )

            tab.create_tabular_tex_code_with_reg_results(
                file=str(produces["tex"]).replace(".tex", ""),
                results=results_dict,
                regressor="treated",
                column_headers=column_headers,
                dep_var_names=dep_var_names,
                version=4,
                gen_pdf=True,
            )

        else:
            tab.create_tabular_tex_code_with_reg_results(
                file=str(produces["tex"]).replace(".tex", ""),
                results=results_dict,
                regressor="treated",
                column_headers=column_headers,
                dep_var_names=dep_var_names,
                version=4,
                gen_pdf=True,
            )

            tab.create_tabular_tex_code_with_reg_results(
                file=str(produces["tex_tabular"]).replace(".tex", ""),
                results=results_dict,
                regressor="treated",
                column_headers=column_headers,
                dep_var_names=dep_var_names,
                version=4,
                gen_pdf=False,
            )
