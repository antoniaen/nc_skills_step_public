"""Sample restrictions used for the analysis."""


def select_sample_for_analysis(data, y_vars, n_years, reform_list):
    """Select the sample used for the analysis.

    Args:
        data (pandas DataFrame): The data set.
        y_vars (list of strings): The dependent variables.
        n_years (int): How many years before and after the cutoff to include.
        reform_list (list of strings): The reforms to include.

    Returns:
        data_copy (pandas DataFrame): The selected sample.

    """
    data_copy = data.copy()
    data_copy = data_copy.query("age > 23").dropna(subset=y_vars)

    # Restrict to the desired reforms.
    data_copy = data_copy[data_copy["country_reform"].isin(reform_list)]

    # Restrict to the desired time window.
    data_copy = data_copy[data_copy["rel_cohort"].isin(range(-n_years, n_years))]

    return data_copy


def select_sample_for_analysis_months_based(data, y_vars, n_months, reform_list):
    """Select the sample used for the analysis.

    Args:
        data (pandas DataFrame): The data set.
        y_vars (list of strings): The dependent variables.
        n_months (int): How many months before and after the cutoff to include.
        reform_list (list of strings): The reforms to include.

    Returns:
        data_copy (pandas DataFrame): The selected sample.

    """
    data_copy = data.copy()
    data_copy = data_copy.query("age > 23").dropna(subset=y_vars)

    # Restrict to the desired reforms.
    data_copy = data_copy[data_copy["country_reform_w_month"].isin(reform_list)]

    # Restrict to the desired time window.
    data_copy = data_copy[data_copy["rel_month"].isin(range(-n_months, n_months))]

    return data_copy


def select_sample_for_placebo_test(data, y_vars, n_years, reform_list, placebo_year):
    """Select the sample used for the placebo test.

    Args:
        data (pandas DataFrame): The data set.
        y_vars (list of strings): The dependent variables.
        number_of_years (int): How many years before and after the cutoff to include.
        reform_list (list of strings): The reforms to include.
        placebo_year (int): Indicates the placebo variable.

    Returns:
        data_copy (pandas DataFrame): The selected sample.

    """
    data_copy = data.copy()
    data_copy = data_copy.query("age > 23").dropna(subset=y_vars)

    # Restrict to the desired reforms.
    data_copy = data_copy[
        data_copy["country_reform_placebo" + str(placebo_year)].isin(reform_list)
    ]

    # Restrict to the desired time window.
    data_copy = data_copy[
        data_copy["rel_placebo_cohort" + str(placebo_year)].isin(
            range(-n_years, n_years),
        )
    ]

    return data_copy


def select_sample_for_robustness_check_wo_piv_cohorts(
    data,
    y_vars,
    n_years,
    reform_list,
):
    """Select the sample used for a robustness check.

    The last untreated cohort is excluded from the sample.

    Args:
        data (pandas DataFrame): The data set.
        y_vars (list of strings): The dependent variables.
        n_years (int): How many years before and after the cutoff to include.
        reform_list (list of strings): The reforms to include.

    Returns:
        data_copy (pandas DataFrame): The selected sample.

    """
    data_copy = data.copy()
    data_copy = data_copy.query("age > 23").dropna(subset=y_vars)

    # Restrict to the desired reforms.
    data_copy = data_copy[data_copy["country_reform"].isin(reform_list)]

    # Restrict to the desired time window.
    data_copy = data_copy[
        data_copy["rel_cohort"].isin([*range(-n_years - 1, -1), *range(0, n_years)])
    ]

    return data_copy


def select_sample_for_robustness_check_wo_age_restriction(
    data,
    y_vars,
    n_years,
    reform_list,
):
    """Select the sample used for a robustness check without restricting age.

    Args:
        data (pandas DataFrame): The data set.
        y_vars (list of strings): The dependent variables.
        n_years (int): How many years before and after the cutoff to include.
        reform_list (list of strings): The reforms to include.

    Returns:
        data_copy (pandas DataFrame): The selected sample.

    """
    data_copy = data.copy()
    data_copy = data_copy.dropna(subset=y_vars)

    # Restrict to the desired reforms.
    data_copy = data_copy[data_copy["country_reform"].isin(reform_list)]

    # Restrict to the desired time window.
    data_copy = data_copy[data_copy["rel_cohort"].isin(range(-n_years, n_years))]

    return data_copy
