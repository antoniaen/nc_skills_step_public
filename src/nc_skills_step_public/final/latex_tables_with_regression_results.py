"""Functions for LaTeX tables with multiple regression results."""

import pylatex as pl


def create_tabular_tex_code_with_descriptives(
    file,
    means,
    results,
    regressor,
    dep_var_names,
    N_dict,
):
    """LaTeX tabular code with descriptive statistics.

    Args:
        file (str): Path to .tex file.
        means (pandas.DataFrame): The means for each group.
        results (dict): Keys are dependent variables and values are OLSResults.
        regressor (str): The regressor.
        dep_var_names (dict): Keys are dependent variables, values are nice labels.
        N_dict (dict): Keys are group names, values are number of observations.

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * 4)
    tabular.add_hline()
    tabular.add_hline()
    tabular.add_row(("", "Control", "Treatment", "Difference", "SE"))
    tabular.add_hline()

    for dep_var in results:
        tabular.add_row(
            (
                dep_var_names[dep_var],
                f"{means.loc[0.0][dep_var]:.2f}",
                f"{means.loc[1.0][dep_var]:.2f}",
                *coef[dep_var],
                *se[dep_var],
            ),
        )

    tabular.add_hline()
    tabular.add_row(("Observations", N_dict[0.0], N_dict[1.0], "", ""))

    tabular.add_hline()
    tabular.add_hline()

    tabular.generate_tex(file)


def create_tabular_tex_code_check_balanced_sample(
    file,
    means,
    results,
    regressor,
    dep_var_names,
    N_dict,
):
    """LaTeX tabular code with descriptive statistics.

    Args:
        file (str): Path to .tex file.
        means (pandas.DataFrame): The means for each group.
        results (dict): Keys are dependent variables and values are OLSResults.
        regressor (str): The regressor.
        dep_var_names (dict): Keys are dependent variables, values are nice labels.
        N_dict (dict): Keys are 0 or 1, values are number of observations.

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * 3)
    tabular.add_hline()
    tabular.add_hline()
    tabular.add_row(("", "Mean", "Treated", "SE"))
    tabular.add_hline()

    for dep_var in results:
        tabular.add_row(
            (
                dep_var_names[dep_var],
                f"{means.loc[dep_var, 0]:.2f}",
                *coef[dep_var],
                *se[dep_var],
            ),
        )

    tabular.add_hline()
    tabular.add_row(("Observations", sum(N_dict.values()), "", ""))

    tabular.add_hline()
    tabular.add_hline()

    tabular.generate_tex(file)


def create_tabular_tex_code_reg_results_show_regressors(
    file,
    results,
    regressor,
    column_headers,
    reg_var_names,
    additional_regressor=None,
    version=1,
):
    """LaTeX table with estimated OLS-coefficients of selected regressor(s).

    The name of the regressors are in the first column, different models in the
    other columns.

    Args:
        file (string): Path to file.
        results (dict): Dictionary where keys are dependent variables and values are
            (multiple) OLSResult objects as list. Lists should have the same length.
        regressor (string): Regressor of interest.
        column_headers (list):  Captions for results columns, e.g. Model 1, Model 2.
        reg_var_names (dict): Keys are regressors, values are the nice labels.
        additional_regressor (string): Additional regressor of interest.
        version (int): 1, 2, 3 or 4. Different table headers (see _multicolumns_header for infos).

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    if additional_regressor is None:
        pass
    else:
        coef_add, se_add, M = _format_coef_se_for_table(
            results,
            additional_regressor,
        )

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * M)
    tabular.add_hline()
    tabular.add_hline()

    tabular = _multicolumns_header(tabular=tabular, version=version)

    tabular.add_row(("", *column_headers))
    tabular.add_row(("", *[f"({i + 1})" for i in range(M)]))
    tabular.add_hline()
    tabular.add_row(("", *[""] * M))

    for dep_var in results:
        # coefficients
        tabular.add_row((reg_var_names[regressor], *coef[dep_var]))
        # standard errors
        tabular.add_row(("", *se[dep_var]))

        if additional_regressor is None:
            pass
        else:
            # coefficients
            tabular.add_row((reg_var_names[additional_regressor], *coef_add[dep_var]))
            # standard errors
            tabular.add_row(("", *se_add[dep_var]))

    tabular.add_hline()
    # Number of observations is identical for all dependent variables within the table.
    tabular.add_row(
        (
            "Observations",
            *[int(results[next(iter(results))][i].nobs) for i in range(M)],
        ),
    )
    tabular.add_hline()
    tabular.add_hline()

    tabular.generate_tex(file)


def create_tabular_tex_code_with_reg_results(
    file,
    results,
    regressor,
    column_headers,
    dep_var_names,
    additional_regressor=None,
    version=1,
    gen_pdf=False,
):
    """LaTeX table with estimated OLS-coefficients of one selected regressor.

    Different dependent variables are in the first column, different models in the
    other columns. The number of observations should be the same for all dependent
    variables. But the number can vary across models.

    Args:
        file (string): Path to file.
        results (dict): Dictionary where keys are dependent variables and values are
            (multiple) OLSResult objects as list. Lists should have the same length.
        regressor (string): Regressor of interest.
        column_headers (list or None):  Captions for results columns, e.g. Model 1, Model 2.
        dep_var_names (dict): Keys are dependent variables, values are the nice labels.
        additional_regressor (string): Additional regressor of interest.
        version (int): 1, 2, 3 or 4. Different column headers (see _multicolumns_header for infos).

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    if additional_regressor is None:
        pass
    else:
        coef_add, se_add, M = _format_coef_se_for_table(
            results,
            additional_regressor,
        )

    if gen_pdf is True:
        doc = pl.Document(file, page_numbers=False)
        doc.packages.append(pl.Package("booktabs"))

        # Begin table.
        table = pl.Table()
        table.append(pl.NoEscape(r"\centering"))

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * M)
    tabular.add_hline()
    tabular.add_hline()

    if version is None:
        pass
    else:
        tabular = _multicolumns_header(tabular=tabular, version=version)

    if column_headers is None:
        pass
    else:
        tabular.add_row(("", *column_headers))

    tabular.add_row(("", *[f"({i + 1})" for i in range(M)]))
    tabular.add_hline()
    tabular.add_row(("Dependent variable", *[""] * M))

    for dep_var in results:
        # coefficients
        tabular.add_row((dep_var_names[dep_var], *coef[dep_var]))
        # standard errors
        tabular.add_row(("", *se[dep_var]))

        if additional_regressor is None:
            pass
        else:
            # coefficients
            tabular.add_row(("", *coef_add[dep_var]))
            # standard errors
            tabular.add_row(("", *se_add[dep_var]))

    tabular.add_hline()
    # Number of observations is identical for all dependent variables within the table.
    tabular.add_row(
        (
            "Observations",
            *[int(results[next(iter(results))][i].nobs) for i in range(M)],
        ),
    )
    tabular.add_hline()
    tabular.add_hline()

    if gen_pdf is True:
        # Append tabular to table.
        table.append(tabular)
        # Append table to document.
        doc.append(table)

        # Generate PDF.
        doc.generate_pdf(clean_tex=False, compiler="pdfLaTeX")

    elif gen_pdf is False:
        tabular.generate_tex(file)


def latex_tabular_multiple_outcomes(
    file,
    results,
    regressor,
    column_headers,
    dep_var_names,
    version=1,
):
    """LaTeX table with estimated OLS-coefficients of one selected regressor.

    Different dependent variables are in the first column, different models in the
    other columns. The number of observations can vary between dependent
    variables and models.

    Args:
        file (string): Path to file.
        results (dict): Dictionary where keys are dependent variables and values are
            (multiple) OLSResult objects as list. Lists should have the same length.
        regressor (string): Regressor of interest.
        column_headers (list):  Captions for results columns, e.g. Model 1, Model 2.
        dep_var_names (dict): Keys are dependent variables, values are the nice labels.
        version (int): 1, 2, 3 or 4. Different column headers (see _multicolumns_header for infos).

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * M)
    tabular.add_hline()
    tabular.add_hline()

    tabular = _multicolumns_header(tabular=tabular, version=version)

    tabular.add_row(("", *column_headers))
    tabular.add_row(("", *[f"({i + 1})" for i in range(M)]))
    tabular.add_hline()
    tabular.add_row(("Dependent variable", *[""] * M))

    for dep_var in results:
        # coefficients
        tabular.add_row((dep_var_names[dep_var], *coef[dep_var]))
        # standard errors
        tabular.add_row(("", *se[dep_var]))
        tabular.add_row(("", *[""] * M))
        # observations
        tabular.add_row(
            (
                pl.NoEscape(r"\textit{Observations}"),
                *[int(results[dep_var][i].nobs) for i in range(M)],
            ),
        )
        tabular.add_hline()

    tabular.add_hline()

    tabular.generate_tex(file)


def create_placebo_test_table(
    file,
    results,
    regressor,
    column_headers,
    dep_var_names,
    gen_pdf=False,
):
    """LaTeX table with estimated OLS-coefficients of one selected regressor.

    Different dependent variables are in the first column.

    Args:
        file (string): Path to file.
        results (dict): Dictionary where keys are dependent variables and values are
            OLSResult objects as list. Lists should have the same length.
        regressor (string): Regressor of interest.
        column_headers (list):  Captions for results columns, e.g. Model 1, Model 2.
        dep_var_names (dict): Keys are dependent variables, values are the nice labels.
        gen_pdf (bool): If True, a PDF is generated, otherwise a .tex file.

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    if gen_pdf is True:
        doc = pl.Document(file, page_numbers=False)
        doc.packages.append(pl.Package("booktabs"))

        # Begin table.
        table = pl.Table()
        table.append(pl.NoEscape(r"\centering"))

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * 3)
    tabular.add_hline()
    tabular.add_hline()
    tabular.add_row(("", *column_headers))
    tabular.add_hline()
    tabular.add_row(("Dependent variable", *[""] * 3))

    for dep_var in results:
        tabular.add_row(
            (
                dep_var_names[dep_var],
                *coef[dep_var],
                *se[dep_var],
                f"{results[dep_var][0].nobs:.0f}",
            ),
        )

    tabular.add_hline()
    tabular.add_hline()

    if gen_pdf is True:
        # Append tabular to table.
        table.append(tabular)
        # Append table to document.
        doc.append(table)

        # Generate PDF.
        doc.generate_pdf(clean_tex=False, compiler="pdfLaTeX")

    elif gen_pdf is False:
        tabular.generate_tex(file)


def create_table_with_optimal_bandwidth(
    file,
    results,
    regressor,
    column_headers,
    dep_var_names,
    h_df,
    with_estimates=True,
    gen_pdf=False,
):
    """LaTeX table with estimated OLS-coefficients of one selected regressor.

    Different dependent variables are in the first column.

    Args:
        file (string): Path to file.
        results (dict): Dictionary wherependent variables and values are
            (multiple) OLSResult objects as list. Lists should have the same length.
        regressor (string): Regressor of interest.
        column_headers (list):  Captions for results columns, e.g. Model 1, Model 2.
        dep_var_names (dict): Keys are dependent variables, values are the nice labels.
        h_df (pandas.DataFrame): Optimal bandwidths.
        with_estimates (bool): If True, coef and se estimates are included in the table.
        gen_pdf (bool): If True, a PDF is generated, otherwise a .tex file.

    Returns:
        .tex file in specified path.

    """
    coef, se, M = _format_coef_se_for_table(results, regressor)

    if gen_pdf is True:
        doc = pl.Document(file, page_numbers=False)
        doc.packages.append(pl.Package("booktabs"))

        # Begin table.
        table = pl.Table()
        table.append(pl.NoEscape(r"\centering"))

    if with_estimates is True:
        C = 4
    elif with_estimates is False:
        C = 2

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * C)
    tabular.add_hline()
    tabular.add_hline()
    tabular.add_row(("", *column_headers))
    tabular.add_hline()
    tabular.add_row(("Dependent variable", *[""] * C))

    if with_estimates is True:
        for dep_var in results:
            tabular.add_row(
                (
                    dep_var_names[dep_var],
                    *coef[dep_var],
                    *se[dep_var],
                    f"{h_df.loc[dep_var, 'h (left)']:.0f}",
                    f"{results[dep_var][0].nobs:.0f}",
                ),
            )
    elif with_estimates is False:
        for dep_var in results:
            tabular.add_row(
                (
                    dep_var_names[dep_var],
                    f"{h_df.loc[dep_var, 'h (left)']:.0f}",
                    f"{results[dep_var][0].nobs:.0f}",
                ),
            )

    tabular.add_hline()
    tabular.add_hline()

    if gen_pdf is True:
        # Append tabular to table.
        table.append(tabular)
        # Append table to document.
        doc.append(table)

        # Generate PDF.
        doc.generate_pdf(clean_tex=False, compiler="pdfLaTeX")

    elif gen_pdf is False:
        tabular.generate_tex(file)


def create_tex_table_p_and_MHT_adjusted_pvalues(
    file,
    results_df,
    column_headers,
    dep_var_names,
    version=1,
    gen_pdf=False,
):
    """LaTeX table with estimated OLS-coefficients of one selected regressor.

    Different dependent variables are in the first column, different models in the
    other columns. The number of observations should be the same for all dependent
    variables. But the number can vary across models.

    Args:
        file (string): Path to file.
        results_df (DataFrame): DataFrame with estimated coefficients, p- and adjusted p-values.
            Index is the dependent variable + model number.
        column_headers (list):  Captions for results columns, e.g. Model 1, Model 2.
        dep_var_names (dict): Keys are dependent variables, values are the nice labels.
        version (int): 1 or 2. Header for 5 and 10 years or 10, 3 and 5 years.
        gen_pdf (bool): If True, a PDF is generated, otherwise a .tex file.

    Returns:
        .tex file in specified path.

    """
    if version == 1:
        M = 5
    elif version == 2:
        M = 7
    elif version == 3:
        M = 8
    elif version == 4:
        M = 7

    if gen_pdf is True:
        doc = pl.Document(file, page_numbers=False)
        doc.packages.append(pl.Package("booktabs"))

        # Begin table.
        table = pl.Table()
        table.append(pl.NoEscape(r"\centering"))

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * M)
    tabular.add_hline()
    tabular.add_hline()

    tabular = _multicolumns_header(tabular=tabular, version=version)

    tabular.add_row(("", *column_headers))
    tabular.add_row(("", *[f"({i + 1})" for i in range(M)]))
    tabular.add_hline()
    tabular.add_row(("Dependent variable", *[""] * M))

    for dep_var in dep_var_names:
        # coefficients
        tabular.add_row(
            (
                dep_var_names[dep_var],
                *[
                    round(results_df.loc[dep_var + "_" + str(i), "params"], 2)
                    for i in range(M)
                ],
            ),
        )
        # p-values
        tabular.add_row(
            (
                "",
                *[
                    "("
                    + f"{results_df.loc[dep_var + '_' + str(i), 'pvalues']:.2f}"
                    + ")"
                    for i in range(M)
                ],
            ),
        )
        # MHT-adjusted p-values
        tabular.add_row(
            (
                "",
                *[
                    "["
                    + f"{results_df.loc[dep_var + '_' + str(i), 'adj_pvalues']:.2f}"
                    + "]"
                    for i in range(M)
                ],
            ),
        )

    tabular.add_hline()
    tabular.add_hline()

    if gen_pdf is True:
        # Append tabular to table.
        table.append(tabular)
        # Append table to document.
        doc.append(table)

        # Generate PDF.
        doc.generate_pdf(clean_tex=False, compiler="pdfLaTeX")

    elif gen_pdf is False:
        tabular.generate_tex(file)


def create_table_with_list_of_correlations(file, corr_dict):
    """Create a LaTeX table with correlations.

    The correlations are between non-cognitive skills and outcomes or characterisitcs.

    Args:
        corr_dict (dict): Dictionary with correlations.

    Returns:
        .tex file in specified path.

    """
    # Begin tabular.
    tabular = pl.Tabular("l" * 2)
    tabular.add_hline()
    tabular.add_hline()
    tabular.add_row(("Non-cognitive skill", "Correlations"))
    tabular.add_hline()

    for key in corr_dict:
        tabular.add_row(key, corr_dict[key])

    tabular.add_hline()
    tabular.add_hline()

    tabular.generate_tex(file)


def _format_coef_se_for_table(results, regressor):
    """Format OLS coefficients and standard errors for one particular regressor.

    Significance stars are added.

    Args:
        results (dict): Dictionary where keys are dependent variables and values are
            (multiple) OLSResult objects as list. lists should have the same length.
        regressor (string): Regressor of interest.

    Returns:
        coef (dict): Dictionary with formatted coefficients (with significance stars).
        se (dict): Dictionary with formatted standard errors (in brackets).
        M (int): Number of different models/specifications stored in results.

    """
    # Get number of models stored in results.
    M = len(list(results.values())[0])

    # Prepare empty containers.
    coef = {dep_var: [None] * M for dep_var in results}
    se = {dep_var: [None] * M for dep_var in results}

    for i in range(M):
        for dep_var in results:
            # Add stars to coefficients.
            coef[dep_var][i] = "{:.2f}".format(
                results[dep_var][i].params[regressor],
            ) + _star_function(results[dep_var][i].pvalues[regressor])
            # Add brackets to standard errors.
            se[dep_var][i] = "(" + f"{results[dep_var][i].bse[regressor]:.2f}" + ")"

    return coef, se, M


def _star_function(p):
    """Create significance stars.

    Args:
        p (float): p-value

    Returns:
        star (string): Corresponding significance stars.

    """
    if round(p, 10) < 0.01:
        star = "***"
    elif round(p, 10) < 0.05:
        star = "**"
    elif round(p, 10) < 0.1:
        star = "*"
    else:
        star = ""
    return star


def _multicolumns_header(tabular, version=1):
    """Create multicolumns header for LaTeX table.

    Args:
        tabular (pylatex tabular): Tabular to build on.
        version (int): 1, 2, 3 or 4.

    Returns:pylatex input.

    """
    tabular = tabular

    if version == 1:
        tabular.add_row(
            (
                "",
                pl.MultiColumn(2, align="c", data="5 years"),
                pl.MultiColumn(3, align="c", data="10 years"),
            ),
        )
        tabular.append(pl.NoEscape(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}"))

    elif version == 2:
        tabular.add_row(
            (
                "",
                pl.MultiColumn(3, align="c", data="10 years"),
                pl.MultiColumn(2, align="c", data="3 years"),
                pl.MultiColumn(2, align="c", data="5 years"),
            ),
        )
        tabular.append(
            pl.NoEscape(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"),
        )

    elif version == 3:
        tabular.add_row(
            (
                "",
                pl.MultiColumn(3, align="c", data="10 years"),
                pl.MultiColumn(2, align="c", data="3 years"),
                pl.MultiColumn(3, align="c", data="5 years"),
            ),
        )
        tabular.append(
            pl.NoEscape(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}"),
        )

    elif version == 4:
        tabular.add_row(
            (
                "",
                pl.MultiColumn(2, align="c", data="5 years"),
                pl.MultiColumn(2, align="c", data="3 years"),
                pl.MultiColumn(3, align="c", data="10 years"),
            ),
        )
        tabular.append(
            pl.NoEscape(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}"),
        )

    return tabular
