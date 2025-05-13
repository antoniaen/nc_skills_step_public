"""Create a table with correlations.

The correlations are between non-cognitive skills and outcomes or characteristics.

"""

import numpy as np
import pandas as pd
import pytask
from scipy.stats import pearsonr

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab

for sample in "full", "ten_years":
    kwargs = {
        "sample": sample,
        "produces": {
            "coef": BLD / "python" / "tables" / f"correlations_{sample}_sample.tex",
            "sign": BLD
            / "python"
            / "tables"
            / f"correlations_{sample}_sample_sign.tex",
        },
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["select_sample_for_analysis.py"],
            "latex_tables": SRC / "final" / "latex_tables_with_regression_results.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=sample, kwargs=kwargs)
    def task_correlation_table(depends_on, sample, produces):
        "Correlation between non-cognitive skills and outcomes or characteristics."
        data = pd.read_pickle(depends_on["data"])

        if sample == "full":
            data_sel = data.copy()
            data_sel = data_sel.query("age > 23")
            # Restrict to the desired countries.
            data_sel = data_sel[
                data_sel["country"].isin(["Bolivia", "Columbia", "Ghana", "Vietnam"])
            ]

        elif sample == "ten_years":
            # Select the sample for the analysis
            data_sel = sel.select_sample_for_analysis(
                data=data,
                y_vars=["country"],
                n_years=10,
                reform_list=gl.reforms_final,
            )

        # Which correlations to calculate.
        dict_which_ones = {
            "agreeableness_av_s_abcorr": ["female", "age"],
            "conscientiousness_av_s_abcorr": ["female", "age"],
            "stability_av_s_abcorr": ["female", "age", "life_sat", "abuse_any_age15"],
            "extraversion_av_s_abcorr": ["female", "age", "life_sat"],
            "openness_av_s_abcorr": ["female", "age"],
            "decision_av_s_abcorr": ["female", "age", "life_sat"],
            "grit_av_s_abcorr": ["conscientiousness_av_s_abcorr"],
            "hostile_av_s_abcorr": ["abuse_any_age15"],
            "patience_binary": ["life_sat"],
            "risk_binary": ["female", "age"],
        }

        # Calculate the correlations
        dict_corr_df = {
            key: data_sel[[key, *value]].corr()
            for key, value in dict_which_ones.items()
        }

        # Calculate the significance levels for the correlations.
        dict_sign_df = {
            key: data_sel[[key, *value]].corr(method=lambda x, y: pearsonr(x, y)[1])
            - np.eye(len(data_sel[[key, *value]].columns))
            for key, value in dict_which_ones.items()
        }

        # Store nice names for the variables.
        nice_names = {
            "female": "Female",
            "age": "Age",
            "life_sat": "Life satisfaction",
            "abuse_any_age15": "Abuse before age 15",
        }
        nice_names.update(gl.nice_variable_names)

        # Extract the correlations into a dictionary.
        final_dict = {
            gl.nice_variable_names[key]: ", ".join(
                f"{nice_names[var]}: {dict_corr_df[key].loc[key, var]:.2f}"
                for var in value
            )
            for key, value in dict_which_ones.items()
        }

        # Extract significance levels for the correlations.
        final_dict_sign = {
            gl.nice_variable_names[key]: ", ".join(
                f"{nice_names[var]}: {dict_sign_df[key].loc[key, var]:.3f}"
                for var in value
            )
            for key, value in dict_which_ones.items()
        }

        # Create a LaTeX table.
        tab.create_table_with_list_of_correlations(
            file=str(produces["coef"]).replace(".tex", ""),
            corr_dict=final_dict,
        )

        tab.create_table_with_list_of_correlations(
            file=str(produces["sign"]).replace(".tex", ""),
            corr_dict=final_dict_sign,
        )
