"""Here are functions to run regressions other than RDD regressions."""

import statsmodels.formula.api as smf


def wage_returns_regression(
    data,
    y_var,
    set_of_regressors1,
    set_of_regressors2,
    set_of_regressors3,
    set_of_regressors4,
    set_of_regressors5,
):
    """Run a wage returns regression.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependen variable.
        set_of_regressors1-5 (list of strings): List of regressors.

    Returns:
        results (list of OLSResults): The regression results.

    """
    regressors = (
        set_of_regressors1
        + set_of_regressors2
        + set_of_regressors3
        + set_of_regressors4
        + set_of_regressors5
    )

    reg_data = data.dropna(subset=[*regressors, y_var]).copy()

    formulas = [None] * 6
    formulas[0] = f"{y_var} ~ " + "+".join(set_of_regressors1)
    formulas[1] = f"{y_var} ~ " + "+".join(set_of_regressors1 + set_of_regressors2)
    formulas[2] = f"{y_var} ~ " + "+".join(set_of_regressors1 + set_of_regressors3)
    formulas[3] = f"{y_var} ~ " + "+".join(set_of_regressors1 + set_of_regressors4)
    formulas[4] = f"{y_var} ~ " + "+".join(set_of_regressors1 + set_of_regressors5)
    formulas[5] = f"{y_var} ~ " + "+".join(regressors)

    models = []
    for i in range(6):
        models.append(smf.ols(formulas[i], data=reg_data))

    results = []
    for i in range(6):
        results.append(
            models[i].fit(cov_type="HC2"),
        )

    return results


def placebo_test(
    data,
    y_var,
    placebo_year,
):
    """Run RDD regression allowing for different slopes before and after the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        placebo_year (int): Indicates the placebo variable.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "placebo",
        "rel_placebo_cohort" + str(placebo_year),
        "country_reform_placebo" + str(placebo_year),
        "siblings_age12",
        y_var,
    ]
    formula = f"{y_var} ~ placebo + country_reform_placebo{placebo_year} + rel_placebo_cohort{placebo_year}:country_reform_placebo{placebo_year} + placebo:rel_placebo_cohort{placebo_year}:country_reform_placebo{placebo_year} + siblings_age12 - 1"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": reg_data[
                "country_reform_placebo" + str(placebo_year) + "_brth_year"
            ],
        },
    )

    return results


def fit_for_RDD_plot(
    data,
    y_var,
    months=False,
    partially_treated=False,
    partially_treated_trend=False,
):
    """Run RDD regression allowing for different slopes before and after the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        months (bool): If True: months instead of years are used as running variable.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.

    Returns:
        results (OLSResults): The regression results.

    """
    if months is False:
        subset = ["treated", "rel_cohort", "country_reform", y_var]
        formula = f"{y_var} ~ treated + country_reform + rel_cohort:country_reform + treated:rel_cohort:country_reform - 1"
        cluster = "country_reform_brth_year"

    elif months is True:
        subset = ["treated_w_month", "rel_month", "country_reform_w_month", y_var]
        formula = f"{y_var} ~ treated_w_month + country_reform_w_month + rel_month:country_reform_w_month + treated_w_month:rel_month:country_reform_w_month - 1"
        cluster = "country_reform_w_month_brth_year"

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform:rel_cohort"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data[cluster]},
    )

    return results


def linear_flexible_trends_single_reform(
    data,
    y_var,
    partially_treated=False,
):
    """Run RDD regression allowing for different slopes before and after the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        partially_treated (bool): If True: indicator for partially treated is added.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = ["treated", "rel_cohort", "country_reform", "siblings_age12", y_var]
    formula = f"{y_var} ~ treated + rel_cohort + treated:rel_cohort + siblings_age12"

    if partially_treated is True:
        formula += " + partially_treated"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_brth_year"]},
    )

    return results
