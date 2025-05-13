"""Predcit wage changes.

The predictions are based on the treatment effects of the reforms and the correlations of skills with wages.

"""

import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import analysis_RDD as reg
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC

skills = {
    "all_skills": ["years_educ"]
    + gl.groups_of_dependent_variables["cogn_skills"]
    + gl.groups_of_dependent_variables["ncogn_skills"]
    + gl.groups_of_dependent_variables["preferences_binary"],
    "non-cognitive_skills": gl.groups_of_dependent_variables["ncogn_skills"]
    + gl.groups_of_dependent_variables["preferences_binary"],
    "cognitive_skills": gl.groups_of_dependent_variables["cogn_skills"],
    "years_of_education": ["years_educ"],
}

for skill_set in skills:
    kwargs = {
        "skill_set": skill_set,
        "produces": BLD / "python" / "other" / f"predicted_wage_change_{skill_set}.tex",
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["analysis_RDD.py", "select_sample_for_analysis.py"],
            "global_info": SRC / "global_info.py",
            "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        },
    )
    @pytask.mark.task(id=skill_set, kwargs=kwargs)
    def task_predict_wage_change(depends_on, skill_set, produces):
        """Use wage correlations and treatment effects to predict wage change."""
        data = pd.read_pickle(depends_on["data"])

        # Get the correlations of skills with wages using ALL skills.
        data_wage_returns = data.query("age > 23").copy()
        data_wage_returns = data_wage_returns[
            data_wage_returns["country"].isin(
                ["Bolivia", "Columbia", "Ghana", "Vietnam"],
            )
        ]
        data_wage_returns = data_wage_returns.dropna(
            subset=[
                *skills["all_skills"],
                "female",
                "age",
                "age2",
                "ln_earnings_h_usd",
            ],
        ).copy()

        formula = "ln_earnings_h_usd ~ female + age + age2 +" + "+".join(
            skills["all_skills"],
        )
        model = smf.ols(formula=formula, data=data_wage_returns)
        results = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": data_wage_returns["country"]},
        )

        # Select skills that are in the current skill set.
        params_list = []
        for skill in skills[skill_set]:
            if results.pvalues[skill] < 0.1:
                params_list.append(results.params[skill])
            else:
                params_list.append(0)

        # Get the effects of the treatment (reform) on skills.
        results_dict = {key: None for key in skills[skill_set]}

        for skill in skills[skill_set]:
            reg_data_5y = sel.select_sample_for_analysis(
                data=data,
                y_vars=[skill],
                n_years=5,
                reform_list=gl.reforms_final,
            )
            results_dict[skill] = reg.linear_flexible_trends(
                data=reg_data_5y,
                y_var=skill,
                reform_type_dummy=False,
                partially_treated=True,
            )

        effect_list = []
        for skill in skills[skill_set]:
            if results_dict[skill].pvalues["treated"] < 0.1:
                effect_list.append(results_dict[skill].params["treated"])
            else:
                effect_list.append(0)

        # Predict wage change.
        delta_wage = np.multiply(effect_list, params_list)

        # Save the result in a tex-document.
        with open(produces, "w") as file:
            file.writelines(
                f"{round(delta_wage.sum(), 3)}",
            )
