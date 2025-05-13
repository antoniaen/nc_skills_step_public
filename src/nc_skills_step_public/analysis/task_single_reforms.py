"""LaTeX tables with multiple regression results from *single reforms* in one table.

Partially treated individuals are included.

"""
import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_other_regressions as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

y_vars = (
    gl.groups_of_dependent_variables["ncogn_skills"]
    + gl.groups_of_dependent_variables["preferences_binary"]
)


@pytask.mark.depends_on(
    {
        "scripts": [
            "analysis_other_regressions.py",
            "select_sample_for_analysis.py",
        ],
        "global_info": SRC / "global_info.py",
        "latex_tables": SRC / "final" / "latex_tables_with_regression_results.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(
    {
        **{
            "txt"
            + y_var
            + reform: BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "single_reforms"
            / f"results_{y_var}_{reform}.txt"
            for y_var in y_vars
            for reform in gl.reforms_final
        },
        **{
            "tex": BLD
            / "python"
            / "tables"
            / "with_partially_treated"
            / "single_reforms"
            / "results_single_reforms.tex",
        },
    },
)
def task_single_reforms_analysis(depends_on, produces):
    """Analysis for single reforms."""
    results_dict = {key: [None] * len(gl.reforms_final) for key in y_vars}
    dep_var_names = {key: gl.nice_variable_names[key] for key in y_vars}

    data = pd.read_pickle(depends_on["data"])

    reg_data_5y = sel.select_sample_for_analysis(
        data=data,
        y_vars=y_vars,
        n_years=5,
        reform_list=gl.reforms_final,
    )

    for _i, reform in enumerate(gl.reforms_final):
        partially_treated = reform == "Vietnam1991"

        for y_var in y_vars:
            reg_data_5y_single_reform = reg_data_5y.query(
                f"country_reform == '{reform}'",
            ).copy()

            results_dict[y_var][_i] = reg.linear_flexible_trends_single_reform(
                data=reg_data_5y_single_reform,
                y_var=y_var,
                partially_treated=partially_treated,
            )

            table = results_dict[y_var][_i].summary().as_text()
            with open(produces["txt" + y_var + reform], "w") as file:
                file.writelines(table)

    tab.create_tabular_tex_code_with_reg_results(
        file=str(produces["tex"]).replace(".tex", ""),
        results=results_dict,
        regressor="treated",
        column_headers=gl.reforms_final,
        dep_var_names=dep_var_names,
        version=None,
    )
