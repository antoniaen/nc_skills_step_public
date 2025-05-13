"""This file executes all Stata do files."""

import pytask

from nc_skills_step_public.config import BLD


@pytask.mark.produces(BLD / "stata" / "create_this_folder.txt")
def task_create_stata_folder(produces):
    open(produces, "w").close()


@pytask.mark.stata(
    script="external_software/fdr_sharpened_qvalues/fdr_sharpened_qvalues_personalized_copy.do",
)
@pytask.mark.depends_on(
    BLD / "python" / "data" / "pvalues" / "common_trend" / "pvalues_combined.xlsx",
)
@pytask.mark.produces(
    BLD / "python" / "data" / "pvalues" / "common_trend" / "qvalues_combined.xlsx",
)
def task_run_fdr_sharpened_qvalues_personalized_copy_do():
    pass
