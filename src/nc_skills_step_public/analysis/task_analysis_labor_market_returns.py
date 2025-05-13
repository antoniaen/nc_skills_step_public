"""Analyze skills' labor market returns within the studied countries."""

import pandas as pd
import pytask
from statsmodels.iolib.summary2 import summary_col

from nc_skills_step_public.analysis import analysis_other_regressions as reg
from nc_skills_step_public.config import BLD, SRC

# Preparation.
set_of_regressors1 = ["years_educ", "female", "age", "age2"]
set_of_regressors2 = ["read_s", "write_s", "num_s"]
set_of_regressors3 = [
    "extraversion_av_s_abcorr",
    "conscientiousness_av_s_abcorr",
    "openness_av_s_abcorr",
    "stability_av_s_abcorr",
    "agreeableness_av_s_abcorr",
]
set_of_regressors4 = ["grit_av_s_abcorr", "decision_av_s_abcorr", "hostile_av_s_abcorr"]
set_of_regressors5 = ["risk_binary", "patience_binary"]


# Define a custom function to indicate the presence of other control variables
def _other_controls_present(model):
    if "age" in model.model.exog_names:
        return "Yes"
    else:
        return "No"


custom_names = {
    "years\\_educ": "Years of education",
    "female": "1 if female",
    "age": "Age",
    "Age2": "Age squared",
    "read\\_s": "Reading",
    "write\\_s": "Writing",
    "num\\_s": "Numeracy",
    "extraversion\\_av\\_s\\_abcorr": "Extraversion",
    "conscientiousness\\_av\\_s\\_abcorr": "Conscientiousness",
    "openness\\_av\\_s\\_abcorr": "Openness to experience",
    "stability\\_av\\_s\\_abcorr": "Emotional stability",
    "agreeableness\\_av\\_s\\_abcorr": "Agreeableness",
    "grit\\_av\\_s\\_abcorr": "Grit",
    "decision\\_av\\_s\\_abcorr": "Decision-making patterns",
    "hostile\\_av\\_s\\_abcorr": "Hostile attribution bias",
    "risk\\_binary": "Risk willingness",
    "patience\\_binary": "Patience",
    # Get tabular not table as output.
    "\\begin{table}\n\\caption{}\n\\label{}\n\\begin{center}\n": "",
    "\n\\end{center}\n\\end{table}\n\\bigskip": "\n\\newline",
    # Get tablenotes in one line.
    "\\newline \n": "",
    # Get nicer table.
    "\\hline": "\\hline\\hline",
}


for y_var in ["ln_earnings_h_usd", "emp"]:
    kwargs = {
        "group": y_var,
        "produces": {
            "main": BLD / "python" / "tables" / f"analysis_returns_{y_var}.tex",
            "main_short": BLD
            / "python"
            / "tables"
            / f"analysis_returns_{y_var}_short_table.tex",
        },
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["analysis_other_regressions.py"],
            "global_info": SRC / "global_info.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=y_var, kwargs=kwargs)
    def task_analysis_returns(depends_on, group, produces):
        """Run a regression analyzing labor market returns in terms of wages and.

        employment.

        """
        data = pd.read_pickle(depends_on["data"])

        data = data.query("age > 23")
        # Restrict to the desired countries.
        data = data[data["country"].isin(["Bolivia", "Columbia", "Ghana", "Vietnam"])]

        results = reg.wage_returns_regression(
            data=data,
            y_var=group,
            set_of_regressors1=set_of_regressors1,
            set_of_regressors2=set_of_regressors2,
            set_of_regressors3=set_of_regressors3,
            set_of_regressors4=set_of_regressors4,
            set_of_regressors5=set_of_regressors5,
        )

        # Long table with all regressors.
        table = summary_col(
            results,
            stars=True,
            float_format="%.2f",
            model_names=[
                "Model 1",
                "Model 2",
                "Model 3",
                "Model 4",
                "Model 5",
                "Model 6",
            ],
            info_dict={
                "N": lambda x: f"{int(x.nobs):d}",
            },
            regressor_order=set_of_regressors1
            + set_of_regressors2
            + set_of_regressors3
            + set_of_regressors4
            + set_of_regressors5,
        )

        latex_table = table.as_latex()

        # Short table only with non-cognitive skills as regressors.
        table_short = summary_col(
            results[2:6],
            stars=True,
            float_format="%.3f",
            model_names=[
                "Model 1",
                "Model 2",
                "Model 3",
                "Model 4",
            ],
            info_dict={
                "N": lambda x: f"{int(x.nobs):d}",
                "Other controls:": _other_controls_present,
            },
            regressor_order=[
                "years_educ",
                *set_of_regressors3,
                *set_of_regressors4,
                *set_of_regressors5,
            ],
            drop_omitted=True,
        )

        latex_table_short = table_short.as_latex()

        for original_name, custom_name in custom_names.items():
            latex_table = latex_table.replace(original_name, custom_name)
            latex_table_short = latex_table_short.replace(original_name, custom_name)

        with open(produces["main"], "w") as file:
            file.writelines(latex_table)
        with open(produces["main_short"], "w") as file:
            file.writelines(latex_table_short)