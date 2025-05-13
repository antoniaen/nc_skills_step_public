"""Analyze the effect on literacy test scores."""

import pandas as pd
import pylatex as pl
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import plausible_values_method as pvm
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_table_literacy_scores as tab

plausible_values = [
    "PVLIT1_s",
    "PVLIT2_s",
    "PVLIT3_s",
    "PVLIT4_s",
    "PVLIT5_s",
    "PVLIT6_s",
    "PVLIT7_s",
    "PVLIT8_s",
    "PVLIT9_s",
    "PVLIT10_s",
]

for type in "fully_only", "partially", "partially_trend":
    kwargs = {
        "type": type,
        "produces": {
            "tex": BLD
            / "python"
            / "tables"
            / "literacy_test_scores"
            / f"results_{type}.tex",
            "pdf": BLD
            / "python"
            / "tables"
            / "literacy_test_scores"
            / f"results_{type}.pdf",
            "tex_tabular": BLD
            / "python"
            / "tables"
            / "literacy_test_scores"
            / f"results_tabular_{type}.tex",
        },
    }

    @pytask.mark.depends_on(
        {
            "scripts": [
                "analysis_RDD.py",
                "select_sample_for_analysis.py",
                "plausible_values_method.py",
            ],
            "global_info": SRC / "global_info.py",
            "latex_tables": SRC / "final" / "latex_table_literacy_scores.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=type, kwargs=kwargs)
    def task_analysis_literacy_test_scores(depends_on, type, produces):
        """Use plausible values methods to analyze the effect on literacy test.

        scores.

        """
        data = pd.read_pickle(depends_on["data"])

        if type == "fully_only":
            reform_list = [
                reform for reform in gl.reforms_final if reform != "Vietnam1991"
            ]
            partially_treated = False
            partially_treated_trend = False

        elif type == "partially":
            reform_list = gl.reforms_final
            partially_treated = True
            partially_treated_trend = False

        elif type == "partially_trend":
            reform_list = gl.reforms_final
            partially_treated = True
            partially_treated_trend = True

        # 3 years
        reg_data_3y = sel.select_sample_for_analysis(
            data=data,
            y_vars=plausible_values,
            n_years=3,
            reform_list=reform_list,
        )
        # 5 years
        reg_data_5y = sel.select_sample_for_analysis(
            data=data,
            y_vars=plausible_values,
            n_years=5,
            reform_list=reform_list,
        )
        # 10 years
        reg_data_10y = sel.select_sample_for_analysis(
            data=data,
            y_vars=plausible_values,
            n_years=10,
            reform_list=reform_list,
        )

        # Run regressions for each of the 10 plausible values.

        model_names = [
            "5Y linear",
            "5Y quadratic",
            "3Y linear",
            "3Y quadratic",
            "10Y linear",
            "10Y quadratic",
            "10Y cubic",
        ]
        coefs = {model: [] for model in model_names}
        sampl_vars = {model: [] for model in model_names}
        coefs2 = {model: [] for model in model_names}
        sampl_vars2 = {model: [] for model in model_names}

        for i in range(1, 11):
            results_3lin = reg.linear_flexible_trends(
                data=reg_data_3y,
                y_var=f"PVLIT{i}_s",
                reform_type_dummy=False,
                partially_treated=partially_treated,
                partially_treated_trend=partially_treated_trend,
            )
            results_3quad = reg.quadratic_flexible_trends(
                data=reg_data_3y,
                y_var=f"PVLIT{i}_s",
                reform_type_dummy=False,
                partially_treated=partially_treated,
                partially_treated_trend=partially_treated_trend,
            )
            results_5lin = reg.linear_flexible_trends(
                data=reg_data_5y,
                y_var=f"PVLIT{i}_s",
                reform_type_dummy=False,
                partially_treated=partially_treated,
                partially_treated_trend=partially_treated_trend,
            )
            results_5quad = reg.quadratic_flexible_trends(
                data=reg_data_5y,
                y_var=f"PVLIT{i}_s",
                reform_type_dummy=False,
                partially_treated=partially_treated,
                partially_treated_trend=partially_treated_trend,
            )
            results_10lin = reg.linear_flexible_trends(
                data=reg_data_10y,
                y_var=f"PVLIT{i}_s",
                reform_type_dummy=False,
                partially_treated=partially_treated,
                partially_treated_trend=partially_treated_trend,
            )
            results_10quad = reg.quadratic_flexible_trends(
                data=reg_data_10y,
                y_var=f"PVLIT{i}_s",
                reform_type_dummy=False,
                partially_treated=partially_treated,
                partially_treated_trend=partially_treated_trend,
            )
            results_10cub = reg.cubic_flexible_trends(
                data=reg_data_10y,
                y_var=f"PVLIT{i}_s",
            )

            results = [
                results_5lin,
                results_5quad,
                results_3lin,
                results_3quad,
                results_10lin,
                results_10quad,
                results_10cub,
            ]

            for j, model in enumerate(model_names):
                coefs[model].append(results[j].params["treated"])
                sampl_vars[model].append(results[j].bse["treated"] ** 2)

                if partially_treated is True:
                    coefs2[model].append(results[j].params["partially_treated"])
                    sampl_vars2[model].append(results[j].bse["partially_treated"] ** 2)

                elif partially_treated is False:
                    pass

        # Store number of observations:
        N = {
            "3 years": results_3lin.nobs,
            "5 years": results_5lin.nobs,
            "10 years": results_10lin.nobs,
        }

        # Treated
        final_coefs, final_std_errors = pvm.plausible_values_method(
            coefs=coefs,
            sampl_vars=sampl_vars,
            model_names=model_names,
            n=10,
        )

        if partially_treated is True:
            # Partially treated
            final_coefs2, final_std_errors2 = pvm.plausible_values_method(
                coefs=coefs2,
                sampl_vars=sampl_vars2,
                model_names=model_names,
                n=10,
            )
        elif partially_treated is False:
            pass

        column_headers = [
            "lin",
            "quad",
            "lin",
            "quad",
            "lin",
            "quad",
            "cub",
        ]

        if partially_treated is True:
            tab.create_tex_table_literacy_test_scores(
                file=str(produces["tex"]).replace(".tex", ""),
                coefs=final_coefs,
                std_errors=final_std_errors,
                nobs=N,
                column_headers=column_headers,
                regressor_names=[
                    "Treated",
                    pl.NoEscape(
                        r"Treated $\times$ Partially treated",
                    ),
                ],
                version=4,
                gen_pdf=True,
                coefs2=final_coefs2,
                std_errors2=final_std_errors2,
            )
            tab.create_tex_table_literacy_test_scores(
                file=str(produces["tex_tabular"]).replace(".tex", ""),
                coefs=final_coefs,
                std_errors=final_std_errors,
                nobs=N,
                column_headers=column_headers,
                regressor_names=[
                    "Treated",
                    pl.NoEscape(
                        r"Treated $\times$ Partially treated",
                    ),
                ],
                version=4,
                gen_pdf=False,
                coefs2=final_coefs2,
                std_errors2=final_std_errors2,
            )

        elif partially_treated is False:
            tab.create_tex_table_literacy_test_scores(
                file=str(produces["tex"]).replace(".tex", ""),
                coefs=final_coefs,
                std_errors=final_std_errors,
                nobs=N,
                column_headers=column_headers,
                regressor_names=["Treated"],
                version=4,
                gen_pdf=True,
            )
            tab.create_tex_table_literacy_test_scores(
                file=str(produces["tex_tabular"]).replace(".tex", ""),
                coefs=final_coefs,
                std_errors=final_std_errors,
                nobs=N,
                column_headers=column_headers,
                regressor_names=["Treated"],
                version=4,
                gen_pdf=False,
            )
