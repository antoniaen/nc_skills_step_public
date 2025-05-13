"""Combine data from excel files with p-values into one excel file.

This combined data is then used for Multiple Hypothesis Testing correction.

"""


import pandas as pd
import pytask

from nc_skills_step_public.config import BLD


@pytask.mark.depends_on(
    {
        "ncogn_skills": BLD
        / "python"
        / "data"
        / "pvalues"
        / "common_trend"
        / "pvalues_ncogn_skills.xlsx",
        "preferences_binary": BLD
        / "python"
        / "data"
        / "pvalues"
        / "common_trend"
        / "pvalues_preferences_binary.xlsx",
    },
)
@pytask.mark.produces(
    BLD / "python" / "data" / "pvalues" / "common_trend" / "pvalues_combined.xlsx",
)
def task_combine_pvalues_for_MHT(depends_on, produces):
    """Combine data from excel files with p-values into one excel file."""
    ncogn_skills = pd.read_excel(depends_on["ncogn_skills"], header=None)
    preferences_binary = pd.read_excel(depends_on["preferences_binary"], header=None)

    combined = pd.concat([ncogn_skills, preferences_binary], axis=0)
    combined.columns = ["dep_vars", "params", "pvalues"]

    # Export DataFrames to Excel using ExcelWriter
    with pd.ExcelWriter(produces) as excel_writer:
        combined.to_excel(excel_writer, sheet_name="Sheet1", index=False)
        combined[["pvalues"]].to_excel(excel_writer, sheet_name="Sheet2", index=False)
