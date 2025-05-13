"""In this file I create latex tables containing descriptive information."""

import pandas as pd
import pytask

from nc_skills_step_public import global_info as gl
from nc_skills_step_public.analysis import select_sample_for_analysis as sel
from nc_skills_step_public.config import BLD, SRC

reform_names = {
    "Ghana1961": "Ghana 1961",
    "Ghana1987": "Ghana 1987",
    "Vietnam1991": "Vietnam 1991",
    "Colombia1991": "Colombia 1991",
    "Bolivia1994": "Bolivia 1994",
}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "step_reforms_final.pkl",
        "select_sample": SRC / "analysis" / "select_sample_for_analysis.py",
        "global_info": SRC / "global_info.py",
    },
)
@pytask.mark.produces(
    {
        "5 years": BLD / "python" / "tables" / "obs_per_reform_5y.tex",
        "10 years": BLD / "python" / "tables" / "obs_per_reform_10y.tex",
        "no missing": BLD / "python" / "tables" / "obs_per_reform_no_missings.tex",
        "final_5y": BLD / "python" / "tables" / "obs_per_reform_final_5y.tex",
        "final_10y": BLD / "python" / "tables" / "obs_per_reform_final_10y.tex",
        "final_3y": BLD / "python" / "tables" / "obs_per_reform_final_3y.tex",
    },
)
def task_tex_table_with_nobs_per_reform(depends_on, produces):
    """Create a tex table with number of observations per reform."""
    data = pd.read_pickle(depends_on["data"])
    data_to_use = data.query("age > 23").copy()

    # 5 years
    data_5y = data_to_use[data_to_use["rel_cohort"].isin(range(-5, 5))].copy()

    table = pd.crosstab(
        data_5y["country_reform"],
        data_5y["treated"],
        margins=True,
    )
    table = table.rename_axis(None)
    table.columns.name = "5 years"
    table = table.rename(columns={0.0: "Control", 1.0: "Treated"})
    # Get rid off "_" in country names.
    table.index = table.index.str.replace("_", " ")
    table.style.format("{:.0f}").to_latex(buf=produces["5 years"], hrules=True)

    # 10 years
    data_10y = data_to_use[data_to_use["rel_cohort"].isin(range(-10, 10))].copy()
    table = pd.crosstab(
        data_10y["country_reform"],
        data_10y["treated"],
        margins=True,
    )
    table = table.rename_axis(None)
    table.columns.name = "10 years"
    table = table.rename(columns={0.0: "Control", 1.0: "Treated"})
    # Get rid off "_" in country names.
    table.index = table.index.str.replace("_", " ")
    table.style.format("{:.0f}").to_latex(buf=produces["10 years"], hrules=True)

    # 10 years + no missing in any dependent variable
    data_no_missing = data_10y.query("age > 23").dropna(
        subset=[
            "years_educ",
            "ln_earnings_h_usd",
            "emp",
            "read_s",
            "write_s",
            "num_s",
            "extraversion_av_s_abcorr",
            "conscientiousness_av_s_abcorr",
            "openness_av_s_abcorr",
            "stability_av_s_abcorr",
            "agreeableness_av_s_abcorr",
            "grit_av_s_abcorr",
            "decision_av_s_abcorr",
            "hostile_av_s_abcorr",
            "risk_s",
            "discount_s",
        ],
    )
    table = pd.crosstab(
        data_no_missing["country_reform"],
        data_no_missing["treated"],
        margins=True,
    )
    table = table.rename_axis(None)
    table.columns.name = "10 years"
    table = table.rename(columns={0.0: "Control", 1.0: "Treated"})
    table.index = table.index.str.replace("_", " ")
    table.style.format("{:.0f}").to_latex(buf=produces["no missing"], hrules=True)

    # Final reforms
    # 5 years
    data_final_5y = sel.select_sample_for_analysis(
        data=data,
        y_vars=["treated"],
        n_years=5,
        reform_list=gl.reforms_final,
    )
    table = pd.crosstab(
        data_final_5y["country_reform"],
        data_final_5y["treated"],
        margins=True,
    )
    table = table.rename_axis(None)
    table.columns.name = "5 years"
    table = table.rename(columns={0.0: "Control", 1.0: "Treated"})
    for name in reform_names:
        table.index = table.index.str.replace(name, reform_names[name])
    table_latex = (
        table.style.format("{:.0f}")
        .to_latex(hrules=True)
        .replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
    )
    with open(produces["final_5y"], "w") as file:
        file.writelines(table_latex)

    # Final reforms - 10 years
    data_final_10y = sel.select_sample_for_analysis(
        data=data,
        y_vars=["treated"],
        n_years=10,
        reform_list=gl.reforms_final,
    )
    table = pd.crosstab(
        data_final_10y["country_reform"],
        data_final_10y["treated"],
        margins=True,
    )
    table = table.rename_axis(None)
    table.columns.name = "10 years"
    table = table.rename(columns={0.0: "Control", 1.0: "Treated"})
    for name in reform_names:
        table.index = table.index.str.replace(name, reform_names[name])
    table_latex = (
        table.style.format("{:.0f}")
        .to_latex(hrules=True)
        .replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
    )
    with open(produces["final_10y"], "w") as file:
        file.writelines(table_latex)

    # Final reforms - 3 years
    data_final_3y = sel.select_sample_for_analysis(
        data=data,
        y_vars=["treated"],
        n_years=3,
        reform_list=gl.reforms_final,
    )
    table = pd.crosstab(
        data_final_3y["country_reform"],
        data_final_3y["treated"],
        margins=True,
    )
    table = table.rename_axis(None)
    table.columns.name = "3 years"
    table = table.rename(columns={0.0: "Control", 1.0: "Treated"})
    for name in reform_names:
        table.index = table.index.str.replace(name, reform_names[name])
    table_latex = (
        table.style.format("{:.0f}")
        .to_latex(hrules=True)
        .replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
    )
    with open(produces["final_3y"], "w") as file:
        file.writelines(table_latex)
