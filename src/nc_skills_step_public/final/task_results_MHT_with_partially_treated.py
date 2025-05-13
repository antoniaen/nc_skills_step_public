"""LaTeX tables with qvalues (MHT correction) for all non-cognitive skills."""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab


@pytask.mark.depends_on(
    {
        "scripts": ["latex_tables_with_regression_results.py"],
        "global_info": SRC / "global_info.py",
        "pvalues_combined": BLD
        / "python"
        / "data"
        / "pvalues"
        / "common_trend"
        / "pvalues_combined.xlsx",
        "qvalues_combined": BLD
        / "python"
        / "data"
        / "pvalues"
        / "common_trend"
        / "qvalues_combined.xlsx",
    },
)
@pytask.mark.produces(
    BLD / "python" / "tables" / "with_partially_treated" / "results_with_qvalues.tex",
)
def task_latex_p_and_qvalues(depends_on, produces):
    """LaTeX tabular with coefficients, p- and q-values."""
    results_df = pd.read_excel(depends_on["pvalues_combined"])
    qvalues = pd.read_excel(depends_on["qvalues_combined"], header=None)

    results_df["adj_pvalues"] = qvalues
    results_df = results_df.set_index("dep_vars")

    column_headers = [
        "lin",
        "quad",
        "lin",
        "quad",
        "lin",
        "quad",
        "cub",
    ]
    dep_vars = (
        gl.groups_of_dependent_variables["ncogn_skills"]
        + gl.groups_of_dependent_variables["preferences_binary"]
    )
    dep_var_names = {dep_var: gl.nice_variable_names[dep_var] for dep_var in dep_vars}

    tab.create_tex_table_p_and_MHT_adjusted_pvalues(
        file=str(produces).replace(".tex", ""),
        results_df=results_df,
        column_headers=column_headers,
        dep_var_names=dep_var_names,
        version=4,
    )