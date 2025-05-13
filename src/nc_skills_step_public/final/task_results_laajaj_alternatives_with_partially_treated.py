"""LaTeX tables with multiple regression results in one table.

Partially treated individuals are included. In this file we use alternative skill
measures inspired by Laajaj et al. (2019).

"""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

for skills_measure in "laajaj_drop", "laajaj_replace":
    kwargs = {
        "skills_measure": skills_measure,
        "produces": {
            "tex": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "Laajaj_alternatives"
            / f"results_with_partially_treated_ncogn_abcorr_{skills_measure}.tex",
            "pdf": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "Laajaj_alternatives"
            / f"results_with_partially_treated_ncogn_abcorr_{skills_measure}.pdf",
            "tex_tabular": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "Laajaj_alternatives"
            / f"results_with_partially_treated_ncogn_abcorr_tabular_{skills_measure}.tex",
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
    @pytask.mark.task(id=skills_measure, kwargs=kwargs)
    def task_latex_results_abcorr_drop_and_replace_skills(
        depends_on,
        skills_measure,
        produces,
    ):
        """Latex table with Big Five results using measures corrected for acquiescence
        bias.

        The skills measure is either the one with dropped or replaced items according to
        Laajaj et al. (2019).

        """
        dep_vars = [
            var.replace("_abcorr", "") + "_" + skills_measure
            for var in gl.groups_of_dependent_variables["ncogn_skills"]
            if var
            not in ["grit_av_s_abcorr", "decision_av_s_abcorr", "hostile_av_s_abcorr"]
        ]
        results_dict = {var: None for var in dep_vars}
        dep_var_names = {
            var.replace("_abcorr", "")
            + "_"
            + skills_measure: gl.nice_variable_names[var]
            for var in gl.groups_of_dependent_variables["ncogn_skills"]
            if var
            not in ["grit_av_s_abcorr", "decision_av_s_abcorr", "hostile_av_s_abcorr"]
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
        / "results_with_partially_treated_ncogn_abcorr_weighted.tex",
        "pdf": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "Laajaj_alternatives"
        / "results_with_partially_treated_ncogn_abcorr_weighted.pdf",
        "tex_tabular": BLD
        / "python"
        / "tables"
        / "with_partially_treated"
        / "Laajaj_alternatives"
        / "results_with_partially_treated_ncogn_abcorr_weighted_tabular.tex",
    },
)
def task_latex_results_abcorr_weighted(depends_on, produces):
    """Latex table with WLS results using measures corrected for acquiescence bias.

    We assign a larger weight to individuals who answered to more items.

    """
    dep_vars = gl.groups_of_dependent_variables["ncogn_skills"]
    results_dict = {var: None for var in dep_vars}
    dep_var_weights = {
        skill: skill.replace("av_s_abcorr", "weight") for skill in dep_vars
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
            weights=dep_var_weights[y_var],
        )
        results_3flex = reg.linear_flexible_trends(
            data=reg_data_3y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
            weights=dep_var_weights[y_var],
        )
        results_5inf = reg.linear_inflexible_trends(
            data=reg_data_5y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
            weights=dep_var_weights[y_var],
        )
        results_5flex = reg.linear_flexible_trends(
            data=reg_data_5y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
            weights=dep_var_weights[y_var],
        )
        results_10inf = reg.linear_inflexible_trends(
            data=reg_data_10y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
            weights=dep_var_weights[y_var],
        )
        results_10flex = reg.linear_flexible_trends(
            data=reg_data_10y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
            weights=dep_var_weights[y_var],
        )
        results_10quad = reg.quadratic_flexible_trends(
            data=reg_data_10y,
            y_var=y_var,
            reform_type_dummy=False,
            partially_treated=True,
            partially_treated_trend=False,
            weights=dep_var_weights[y_var],
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
        dep_var_names=gl.nice_variable_names,
        version=2,
        gen_pdf=True,
    )

    tab.create_tabular_tex_code_with_reg_results(
        file=str(produces["tex_tabular"]).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=column_headers,
        dep_var_names=gl.nice_variable_names,
        version=2,
        gen_pdf=False,
    )
