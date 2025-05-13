import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.data_management import prepare_merged_data as prep


@pytask.mark.depends_on(
    {
        "scripts": ["prepare_merged_data.py"],
        "global_info": SRC / "global_info.py",
        "step_reforms": BLD / "python" / "data" / "STEP_and_reforms.pkl",
    },
)
@pytask.mark.produces(
    {
        "pkl": BLD / "python" / "data" / "step_reforms_final.pkl",
        "csv": BLD / "python" / "data" / "step_reforms_final.csv",
    },
)
def task_prepare_merged_data(depends_on, produces):
    """Prepare merged data by adding variables."""
    data = pd.read_pickle(depends_on["step_reforms"])

    data = prep.create_treatment_indicator(data=data, var_name="treated")
    data = prep.create_treatment_indicator_w_month(data=data)
    data = prep.create_individuals_relevant_reform(data=data, var_name="country_reform")
    data = prep.create_individuals_relevant_reform_months_based(
        data=data,
        var_name="country_reform_w_month",
    )
    data = prep.create_partially_treated_indicator(data=data)
    data = prep.create_relative_cohort(data=data)
    data = prep.create_relative_month(data=data)
    # Add placebo variables.
    data = prep.create_relative_placebo_cohort(data=data)

    for year in gl.placebo_years:
        data = prep.create_treatment_indicator(
            data=data,
            var_name="placebo" + gl.placebo_years[year],
            placebo=year,
        )
        data = prep.create_individuals_relevant_reform(
            data=data,
            var_name="country_reform_placebo" + gl.placebo_years[year],
            placebo=year,
        )
        data = prep.create_partially_treated_placebo_indicator(
            data=data,
            placebo_number=gl.placebo_years[year],
        )

    data.to_pickle(produces["pkl"])
    data.to_csv(produces["csv"])
