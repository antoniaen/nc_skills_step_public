"""This file contains the RDD regression functions for our main analysis."""

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices


def linear_inflexible_trends(
    data,
    y_var,
    reform_type_dummy,
    partially_treated=False,
    partially_treated_trend=False,
    weights=None,
):
    """Run RDD regression with country-reform-fe and -birth-cohort-trends.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        reform_type_dummy (bool): If True: indicator for unsuccessful reforms is added.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.
        weights (string): Weights for WLS.

    Returns:
        results (OLSResults or WLSResults): The regression results.

    """
    if reform_type_dummy is False:
        subset = [
            "treated",
            "rel_cohort",
            "country_reform",
            "siblings_age12",
            y_var,
        ]
        formula = f"{y_var} ~ treated + country_reform + rel_cohort:country_reform + siblings_age12 - 1"
    elif reform_type_dummy is True:
        subset = [
            "treated",
            "rel_cohort",
            "country_reform",
            "unsuccessful_reform",
            "siblings_age12",
            y_var,
        ]
        formula = f"{y_var} ~ treated + treated:unsuccessful_reform + country_reform + rel_cohort:country_reform + siblings_age12 - 1"
    else:
        print("Argument reform_type_dummy must be either True or False")

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform:rel_cohort"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    if weights is None:
        model = smf.ols(formula=formula, data=reg_data)

    elif weights is not None:
        y, X = dmatrices(formula, data=reg_data, return_type="dataframe")
        model = sm.WLS(y, X, weights=reg_data[weights])

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_brth_year"]},
    )

    return results


def linear_flexible_trends(
    data,
    y_var,
    reform_type_dummy,
    partially_treated=False,
    partially_treated_trend=False,
    weights=None,
):
    """Run RDD regression allowing for different slopes before and after the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        reform_type_dummy (bool): If True: indicator for unsuccessful reforms is added.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.
        weights (string): Weights for WLS.

    Returns:
        results (OLSResults or WLSResults): The regression results.

    """
    if reform_type_dummy is False:
        subset = [
            "treated",
            "rel_cohort",
            "country_reform",
            "siblings_age12",
            y_var,
        ]
        formula = f"{y_var} ~ treated + country_reform + rel_cohort:country_reform + treated:rel_cohort:country_reform + siblings_age12 - 1"
    elif reform_type_dummy is True:
        subset = [
            "treated",
            "rel_cohort",
            "country_reform",
            "unsuccessful_reform",
            "siblings_age12",
            y_var,
        ]
        formula = f"{y_var} ~ treated + treated:unsuccessful_reform + country_reform + rel_cohort:country_reform + treated:rel_cohort:country_reform + siblings_age12 - 1"
    else:
        print("Argument reform_type_dummy must be either True or False")

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform:rel_cohort"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    if weights is None:
        model = smf.ols(formula=formula, data=reg_data)

    elif weights is not None:
        y, X = dmatrices(formula, data=reg_data, return_type="dataframe")
        model = sm.WLS(y, X, weights=reg_data[weights])

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_brth_year"]},
    )

    return results


def quadratic_flexible_trends(
    data,
    y_var,
    reform_type_dummy,
    partially_treated=False,
    partially_treated_trend=False,
    weights=None,
):
    """Run RDD regression with quadratic age trends and different slopes at the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.
        reform_type_dummy (bool): If True: indicator for unsuccessful reforms is added.
        partially_treated (bool): If True: indicator for partially treated is added.
        partially_treated_trend (bool): If True: separate trend for partially treated is added.
        weights (string): Weights for WLS.

    Returns:
        results (OLSResults or WLSResults): The regression results.

    """
    if reform_type_dummy is False:
        subset = [
            "treated",
            "rel_cohort",
            "rel_cohort2",
            "country_reform",
            "siblings_age12",
            y_var,
        ]
        formula = (
            f"{y_var} ~ "
            "treated + "
            "country_reform + "
            "rel_cohort:country_reform + "
            "rel_cohort2:country_reform + "
            "treated:rel_cohort:country_reform + "
            "treated:rel_cohort2:country_reform + "
            "siblings_age12 - "
            "1"
        )
    elif reform_type_dummy is True:
        subset = [
            "treated",
            "rel_cohort",
            "rel_cohort2",
            "country_reform",
            "unsuccessful_reform",
            "siblings_age12",
            y_var,
        ]
        formula = (
            f"{y_var} ~ "
            "treated + "
            "treated:unsuccessful_reform + "
            "country_reform + "
            "rel_cohort:country_reform + "
            "rel_cohort2:country_reform + "
            "treated:rel_cohort:country_reform + "
            "treated:rel_cohort2:country_reform + "
            "siblings_age12  - "
            "1"
        )
    else:
        print("Argument reform_type_dummy must be either True or False")

    if partially_treated is True:
        formula += " + partially_treated"

    if partially_treated_trend is True:
        formula += " + partially_treated:country_reform:rel_cohort"

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    if weights is None:
        model = smf.ols(formula=formula, data=reg_data)

    elif weights is not None:
        y, X = dmatrices(formula, data=reg_data, return_type="dataframe")
        model = sm.WLS(y, X, weights=reg_data[weights])

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_brth_year"]},
    )

    return results


def cubic_flexible_trends(
    data,
    y_var,
):
    """Run RDD regression with cubic age trends and different slopes at the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "treated",
        "rel_cohort",
        "rel_cohort2",
        "rel_cohort3",
        "country_reform",
        "siblings_age12",
        y_var,
    ]
    formula = (
        f"{y_var} ~ "
        "treated + "
        "partially_treated + "
        "country_reform + "
        "rel_cohort:country_reform + "
        "rel_cohort2:country_reform + "
        "rel_cohort3:country_reform + "
        "treated:rel_cohort:country_reform + "
        "treated:rel_cohort2:country_reform + "
        "treated:rel_cohort3:country_reform + "
        "siblings_age12 - "
        "1"
    )

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_brth_year"]},
    )

    return results


def quartic_flexible_trends(
    data,
    y_var,
):
    """Run RDD regression with quartic age trends and different slopes at the cutoff.

    Args:
        data (pandas DataFrame): The data set.
        y_var (string): Dependent variable.

    Returns:
        results (OLSResults): The regression results.

    """
    subset = [
        "treated",
        "rel_cohort",
        "rel_cohort2",
        "rel_cohort3",
        "rel_cohort4",
        "country_reform",
        "siblings_age12",
        y_var,
    ]
    formula = (
        f"{y_var} ~ "
        "treated + "
        "partially_treated + "
        "country_reform + "
        "rel_cohort:country_reform + "
        "rel_cohort2:country_reform + "
        "rel_cohort3:country_reform + "
        "rel_cohort4:country_reform + "
        "treated:rel_cohort:country_reform + "
        "treated:rel_cohort2:country_reform + "
        "treated:rel_cohort3:country_reform + "
        "treated:rel_cohort4:country_reform + "
        "siblings_age12 - "
        "1"
    )

    # Drop rows with missing values.
    reg_data = data.dropna(subset=subset).copy()

    model = smf.ols(formula=formula, data=reg_data)

    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_data["country_reform_brth_year"]},
    )

    return results
