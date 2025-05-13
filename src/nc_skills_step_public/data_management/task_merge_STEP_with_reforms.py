"""Merge STEP data with reform data."""

import pandas as pd
import pytask

from nc_skills_step_public.config import BLD, SRC


@pytask.mark.depends_on(
    {
        "STEP": BLD / "python" / "data" / "STEP_data_clean.pkl",
        "reforms": SRC / "data" / "reforms.xlsx",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "STEP_and_reforms.pkl")
def task_merge_with_reforms(depends_on, produces):
    """Merge STEP data with reform data."""
    reforms = pd.read_excel(depends_on["reforms"])

    # Reform-data from long to wide
    reforms["reform"] = reforms.groupby("country").cumcount() + 1
    reshaped_reforms = reforms.pivot(index="country", columns="reform")
    # Flatten the multi-level column index
    reshaped_reforms.columns = [
        f"{col[0]}_reform{col[1]}" for col in reshaped_reforms.columns
    ]
    reshaped_reforms = reshaped_reforms.reset_index()

    step = pd.read_pickle(depends_on["STEP"])

    step_reforms = step.join(
        reshaped_reforms.set_index("country"),
        on="country",
    )

    step_reforms.to_pickle(produces)
