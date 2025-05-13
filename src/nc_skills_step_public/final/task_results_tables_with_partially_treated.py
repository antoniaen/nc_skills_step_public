"""LaTeX tables with multiple regression results in one table.

Partially treated individuals are included. This file produces our main results.

"""

import pandas as pd
import pylatex as pl
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

for trend in "common_trend", "separate_trends":
    for group in gl.groups_of_dependent_variables:
        kwargs = {
            "trend": trend,
            "group": group,
            "produces": {
                "tex": BLD
                / "python"
                / "tables"
                / "with_partially_treated"
                / trend
                / f"results_with_partially_treated_{group}.tex",
                "pdf": BLD
                / "python"
                / "tables"
                / "with_partially_treated"
                / trend
                / f"results_with_partially_treated_{group}.pdf",
                "tex_tabular": BLD
                / "python"
                / "tables"
                / "with_partially_treated"
                / trend
                / f"results_with_partially_treated_{group}_tabular.tex",
                "results_df": BLD
                / "python"
                / "data"
                / "pvalues"
                / trend
                / f"pvalues_{group}.xlsx",
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
        @pytask.mark.task(id=trend + "_" + group, kwargs=kwargs)
        def task_latex_regression_results_with_partially_treated(
            depends_on,
            trend,
            group,
            produces,
        ):
            """Latex tabular with regression results."""
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

            if trend == "common_trend":
                trend_boolean = False
            elif trend == "separate_trends":
                trend_boolean = True

            for y_var in dep_vars:
                results_3lin = reg.linear_flexible_trends(
                    data=reg_data_3y,
                    y_var=y_var,
                    reform_type_dummy=False,
                    partially_treated=True,
                    partially_treated_trend=trend_boolean,
                )
                results_3quad = reg.quadratic_flexible_trends(
                    data=reg_data_3y,
                    y_var=y_var,
                    reform_type_dummy=False,
                    partially_treated=True,
                    partially_treated_trend=trend_boolean,
                )
                results_5lin = reg.linear_flexible_trends(
                    data=reg_data_5y,
                    y_var=y_var,
                    reform_type_dummy=False,
                    partially_treated=True,
                    partially_treated_trend=trend_boolean,
                )
                results_5quad = reg.quadratic_flexible_trends(
                    data=reg_data_5y,
                    y_var=y_var,
                    reform_type_dummy=False,
                    partially_treated=True,
                    partially_treated_trend=trend_boolean,
                )
                results_10lin = reg.linear_flexible_trends(
                    data=reg_data_10y,
                    y_var=y_var,
                    reform_type_dummy=False,
                    partially_treated=True,
                    partially_treated_trend=trend_boolean,
                )
                results_10quad = reg.quadratic_flexible_trends(
                    data=reg_data_10y,
                    y_var=y_var,
                    reform_type_dummy=False,
                    partially_treated=True,
                    partially_treated_trend=trend_boolean,
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

            # Export coeffs and p-values for Multiple Hypothesis Testing correction.
            results_df = pd.DataFrame()

            for key in results_dict:
                for i in range(len(results_dict[key])):
                    results_df.loc[key + "_" + str(i), "params"] = results_dict[key][
                        i
                    ].params["treated"]
                    results_df.loc[key + "_" + str(i), "pvalues"] = results_dict[key][
                        i
                    ].pvalues["treated"]

            results_df.to_excel(produces["results_df"], index=True, header=False)
