"""LaTeX tables with multiple regression results in one table.

Partially treated individuals are included. In this file we use skill measures which are
not corrected for acquiescence bias.

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
    {
        "tex": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "Laajaj_alternatives"
        / "results_with_partially_treated_ncogn_not_abcorr.tex",
        "pdf": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "Laajaj_alternatives"
        / "results_with_partially_treated_ncogn_not_abcorr.pdf",
        "tex_tabular": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "Laajaj_alternatives"
        / "results_with_partially_treated_ncogn_not_abcorr_tabular.tex",
    },
)
def task_latex_results_not_abcorr(depends_on, produces):
    """Latex table with results using measures not!

    corrected for acquiescence bias.

    """
    dep_vars = [
        var.replace("_abcorr", "")
        for var in gl.groups_of_dependent_variables["ncogn_skills"]
    ]
    results_dict = {var: None for var in dep_vars}
    dep_var_names = {
        var.replace("_abcorr", ""): gl.nice_variable_names[var]
        for var in gl.groups_of_dependent_variables["ncogn_skills"]
    }

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
        results_3inf = reg.linear_inflexible_trends(
            data=reg_data_3y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_3flex = reg.linear_flexible_trends(
            data=reg_data_3y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_5inf = reg.linear_inflexible_trends(
            data=reg_data_5y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_5flex = reg.linear_flexible_trends(
            data=reg_data_5y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_10inf = reg.linear_inflexible_trends(
            data=reg_data_10y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
        )
        results_10flex = reg.linear_flexible_trends(
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

        results_dict[y_var] = [
            results_10inf,
            results_10flex,
            results_10quad,
            results_3inf,
            results_3flex,
            results_5inf,
            results_5flex,
        ]

    column_headers = [
        "inflex.",
        "flex.",
        "quad.",
        "inflex.",
        "flex.",
        "inflex.",
        "flex.",
    ]

    tab.create_tabular_tex_code_with_reg_results(
        file=str(produces["tex"]).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=column_headers,
        dep_var_names=dep_var_names,
        version=2,
        gen_pdf=True,
    )

    tab.create_tabular_tex_code_with_reg_results(
        file=str(produces["tex_tabular"]).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=column_headers,
        dep_var_names=dep_var_names,
        version=2,
        gen_pdf=False,
    )
