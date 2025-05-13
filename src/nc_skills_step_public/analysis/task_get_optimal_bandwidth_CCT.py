"""Estimate the optimal bandwidth.

Package is based on Calonico, Cattaneo, and Titiunik (2014).

"""

import pandas as pd
import pytask
import rdrobust as rdr

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.config import BLD, SRC


@pytask.mark.depends_on(
    {
        "global_info": SRC / "global_info.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "optimal_bandwidth_CCT.pkl")
def task_optimal_bandwidth_CCT(depends_on, produces):
    """Estimate the optimal bandwidth."""
    data = pd.read_pickle(depends_on["data"])

    data = data.query("age > 23")

    # Restrict to the desired reforms.
    data = data[data["country_reform"].isin(gl.reforms_final)]

    # Create dummies for each reform since RDROBUST can not deal with strings.
    dummies = pd.get_dummies(data["country_reform"], prefix="cr").astype(int)

    # Concatenate the indicator variables with the original DataFrame
    data_with_dummies = pd.concat([data, dummies], axis=1)

    # Prepare the final data set.
    h_df = pd.DataFrame(columns=["h (left)", "h (right)"])

    for y_var in gl.dependent_variables:
        out = rdr.rdbwselect(
            y=data_with_dummies[y_var],
            x=data_with_dummies["rel_cohort"],
            p=1,
            kernel="uniform",
            bwselect="mserd",
            cluster=data_with_dummies["country_reform_brth_year"],
            covs=data_with_dummies[
                [
                    "cr_Bolivia1994",
                    "cr_Colombia1991",
                    "cr_Vietnam1991",
                    "cr_Ghana1961",
                    "cr_Ghana1987",
                    "siblings_age12",
                    "partially_treated",
                ]
            ],
        )
        opt_h = out.bws[["h (left)", "h (right)"]].values.round(0)
        h_df.loc[y_var, :] = opt_h

    h_df.to_pickle(produces)
