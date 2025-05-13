import statsmodels.formula.api as smf


def linear_inflexible_trends_w_month(
    data,
    y_var,
    partially_treated=False,
    partially_treated_trend=False,
):
    """Run RDD regression with country-reform-fe and -birth-month-trends.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "treated_w_month",
        "rel_month",
        "country_reform_w_month",
        "siblings_age12",
        y_var,
    ]
    formula = f"{y_var} ~ treated_w_month + country_reform_w_month + rel_month:country_reform_w_month + siblings_age12 - 1"

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform_w_month:rel_month"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_w_month_brth_year"]},
    )

    return results


def linear_flexible_trends_w_month(
    data,
    y_var,
    partially_treated=False,
    partially_treated_trend=False,
):
    """Run RDD regression allowing for different slopes before and after the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "treated_w_month",
        "rel_month",
        "country_reform_w_month",
        "siblings_age12",
        y_var,
    ]
    formula = (
        f"{y_var} ~ treated_w_month +"
        "country_reform_w_month + rel_month:country_reform_w_month + "
        "treated_w_month:rel_month:country_reform_w_month + siblings_age12 - 1"
    )

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform_w_month:rel_month"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_w_month_brth_year"]},
    )

    return results


def quadratic_inflexible_trends_w_month(
    data,
    y_var,
    partially_treated=False,
    partially_treated_trend=False,
):
    """Run RDD regression with quadratic age trends and the same slope at the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "treated_w_month",
        "rel_month",
        "rel_month2",
        "country_reform_w_month",
        "siblings_age12",
        y_var,
    ]
    formula = (
        f"{y_var} ~ "
        "treated_w_month + "
        "country_reform_w_month + "
        "rel_month:country_reform_w_month + "
        "rel_month2:country_reform_w_month + "
        "siblings_age12 - "
        "1"
    )

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform_w_month:rel_month"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_w_month_brth_year"]},
    )

    return results


def quadratic_flexible_trends_w_month(
    data,
    y_var,
    partially_treated=False,
    partially_treated_trend=False,
):
    """Run RDD regression with quadratic age trends and different slopes at the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "treated_w_month",
        "rel_month",
        "rel_month2",
        "country_reform_w_month",
        "siblings_age12",
        y_var,
    ]
    formula = (
        f"{y_var} ~ "
        "treated_w_month + "
        "country_reform_w_month + "
        "rel_month:country_reform_w_month + "
        "rel_month2:country_reform_w_month + "
        "treated_w_month:rel_month:country_reform_w_month + "
        "treated_w_month:rel_month2:country_reform_w_month + "
        "siblings_age12 - "
        "1"
    )

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform_w_month:rel_month"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_w_month_brth_year"]},
    )

    return results
