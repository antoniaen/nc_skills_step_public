"""Descriptive statistics."""

import pandas as pd
import pytask
import statsmodels.formula.api as smf

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC
from nc_skills_step_public.final import latex_tables_with_regression_results as tab


@pytask.mark.depends_on(
    {
        "scripts": ["latex_tables_with_regression_results.py"],
        "global_info": SRC / "global_info.py",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
    },
)
@pytask.mark.produces(
    {
        "all_simple": BLD / "python" / "tables" / "descriptive_statistics.tex",
        "all_check": BLD / "python" / "tables" / "balanced_sample.tex",
        "years_educ": BLD
        / "python"
        / "tables"
        / "descriptive_statistics_years_educ.tex",
    },
)
def task_descriptive_statistics(depends_on, produces):
    """TeX tabular code with descriptive statistics."""
    data = pd.read_pickle(depends_on["data"])

    variables_to_check = {
        "female": "Female",
        "age_strt_school": "Age at school start",
        "ses_age15": "SES at age 15",
        "parents_info_school": "Parental involvement in educ.",
        "worked_age15": "Worked before age 15",
        "siblings_age12": "Siblings at age 12",
    }

    variables_to_control = {
        "age": "Age",
        "years_educ": "Years of education",
    }

    data_5y = sel.select_sample_for_analysis(
        data=data,
        y_vars=(variables_to_control | variables_to_check).keys(),
        n_years=5,
        reform_list=gl.reforms_final,
    )

    N_dict = data_5y["treated"].value_counts().to_dict()

    # Descriptive statistics - Differences in means
    means_df = (
        data_5y.groupby("treated")[
            list((variables_to_control | variables_to_check).keys())
        ]
        .mean()
        .round(2)
    )

    results = {var: None for var in variables_to_control | variables_to_check}
    for var in variables_to_control | variables_to_check:
        model = smf.ols(f"{var} ~ treated", data=data_5y)
        results[var] = [
            model.fit(
                cov_type="cluster",
                cov_kwds={"groups": data_5y["country_reform_brth_year"]},
            ),
        ]

    tab.create_tabular_tex_code_with_descriptives(
        file=str(produces["all_simple"]).replace(".tex", ""),
        means=means_df,
        results=results,
        regressor="treated",
        dep_var_names=variables_to_control | variables_to_check,
        N_dict=N_dict,
    )

    # Balanced sample check
    means_overall_df = (
        data_5y[list(variables_to_check.keys())].mean().round(2).to_frame()
    )

    results2 = {var: None for var in variables_to_check}
    for var in variables_to_check:
        model2 = smf.ols(
            f"{var} ~ treated + country_reform + rel_cohort:country_reform + treated:rel_cohort:country_reform + partially_treated - 1",
            data=data_5y,
        )
        results2[var] = [
            model2.fit(
                cov_type="cluster",
                cov_kwds={"groups": data_5y["country_reform_brth_year"]},
            ),
        ]

    tab.create_tabular_tex_code_check_balanced_sample(
        file=str(produces["all_check"]).replace(".tex", ""),
        means=means_overall_df,
        results=results2,
        regressor="treated",
        dep_var_names=variables_to_check,
        N_dict=N_dict,
    )

    # Years of education per reform
    means_per_reform = (
        data_5y.groupby(["treated", "country_reform"])["years_educ"].mean().unstack().T
    )
    means_per_reform = means_per_reform.rename_axis(None)
    means_per_reform.columns.name = ""
    means_per_reform = means_per_reform.rename(
        columns={0.0: "Control", 1.0: "Treatment"},
    )

    reform_names = {
        "Ghana1961": "Ghana 1961",
        "Ghana1987": "Ghana 1987",
        "Vietnam1991": "Vietnam 1991",
        "Colombia1991": "Colombia 1991",
        "Bolivia1994": "Bolivia 1994",
    }
    for name in reform_names:
        means_per_reform.index = means_per_reform.index.str.replace(
            name,
            reform_names[name],
        )

    latex = (
        means_per_reform.style.format("{:.2f}")
        .to_latex(
            hrules=True,
        )
        .replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
    )
    with open(produces["years_educ"], "w") as file:
        file.writelines(latex)
