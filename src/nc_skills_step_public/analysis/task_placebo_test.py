"""Placebo test."""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_other_regressions as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

# Placebo Test 1: Shifting all pivotal cohorts.
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
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(
    {
        **{
            "txt"
            + y_var
            + i: BLD
            / "python"
            / "tables"
            / "placebo_test"
            / f"results_{y_var}_placebo_test_{i}.txt"
            for y_var in y_vars
            for i in gl.placebo_years.values()
        },
        **{"tex": BLD / "python" / "tables" / "placebo_test" / "placebo_test.tex"},
        **{
            "tex_multiple": BLD
            / "python"
            / "tables"
            / "placebo_test"
            / "multiple_placebo_test.tex",
        },
    },
)
def task_placebo_test(depends_on, produces):
    """Placebo test."""
    results_dict = {key: [None] * len(gl.placebo_years) for key in y_vars}
    dep_var_names = {key: gl.nice_variable_names[key] for key in y_vars}

    data = pd.read_pickle(depends_on["data"])

    for i in gl.placebo_years.values():
        reg_data_5y = sel.select_sample_for_placebo_test(
            data=data,
            y_vars=y_vars,
            n_years=5,
            # Desired reforms
            reform_list=gl.reforms_final,
            # Number of years shifted from true pivotal cohort.
            placebo_year=i,
        )

        for y_var in y_vars:
            # Rename the placebo-"treated" indicator to use the function creating a latex table.
            reg_data_5y = reg_data_5y.rename(columns={"placebo" + i: "placebo"})

            results_dict[y_var][eval(i)] = reg.placebo_test(
                data=reg_data_5y,
                y_var=y_var,
                placebo_year=i,
            )

            table = results_dict[y_var][eval(i)].summary().as_text()
            with open(produces["txt" + y_var + i], "w") as file:
                file.writelines(table)

    placebo5_dict = {key: [value[-1]] for key, value in results_dict.items()}
    tab.create_placebo_test_table(
        file=str(produces["tex"]).replace(".tex", ""),
        results=placebo5_dict,
        regressor="placebo",
        column_headers=["Placebo", "SE", "N"],
        dep_var_names=dep_var_names,
    )

    tab.create_tabular_tex_code_with_reg_results(
        file=str(produces["tex_multiple"]).replace(".tex", ""),
        results=results_dict,
        regressor="placebo",
        column_headers=["shift -6", "shift -5", "shift +5", "shift +6", "shift +7"],
        dep_var_names=dep_var_names,
        version=None,
    )
