"""Functions for LaTeX tables with results from literacy test scores analysis."""

import pylatex as pl


def create_tex_table_literacy_test_scores(
    file,
    coefs,
    std_errors,
    nobs,
    column_headers,
    regressor_names,
    version=1,
    gen_pdf=True,
    coefs2=None,
    std_errors2=None,
):
    """Latex table (plus compilation) with literacy test scores results.

    Args:
        file (string): Path to file.
        coefs (dict): Dictionary with coefficients.
        std_errors (dict): Dictionary with standard errors.
        nobs (dict): Dictionary with number of observations (keys are "5 years" and "10 years").
        column_headers (list):  Captions for results columns, e.g. Model 1, Model 2.
        regressor_names (list): List with names for regressors.
        version (int): 1, 2, 3 or 4. Different table headers (see _multicolumns_header for infos).
        gen_pdf (bool): If True, generate PDF. If False, generate tabular code only.
        coefs2 (dict): Dictionary with coefficients for additional independent variable.
        std_errors2 (dict): Dictionary with standard errors for additional independent variable.

    Returns:
            .tex and .pdf file in specified path.

    """
    if version == 1:
        M = 5
    elif version == 2:
        M = 7
    elif version == 3:
        M = 8
    elif version == 4:
        M = 7

    # Format coefficients and standard errors.
    coefs_rounded, std_errors_rounded = _format_coefs_and_ses(coefs, std_errors)

    if coefs2 is not None:
        coefs_rounded2, std_errors_rounded2 = _format_coefs_and_ses(coefs2, std_errors2)
    elif coefs2 is None:
        pass

    if gen_pdf is True:
        doc = pl.Document(file, page_numbers=False)
        doc.packages.append(pl.Package("booktabs"))

        # Begin table.
        table = pl.Table()
        table.append(pl.NoEscape(r"\centering"))

    # Begin tabular.
    tabular = pl.Tabular("l" + "c" * len(column_headers))
    tabular.add_hline()
    tabular.add_hline()

    tabular = _multicolumns_header(tabular=tabular, version=version)

    tabular.add_row(("", *column_headers))
    tabular.add_row(("", *[f"({i + 1})" for i in range(M)]))
    tabular.add_hline()
    tabular.add_row("", *[""] * M)
    tabular.add_row(regressor_names[0], *coefs_rounded.values())
    tabular.add_row("", *std_errors_rounded.values())

    if coefs2 is not None:
        tabular.add_row(regressor_names[1], *coefs_rounded2.values())
        tabular.add_row("", *std_errors_rounded2.values())
    elif coefs2 is None:
        pass

    tabular.add_hline()

    if version == 1:
        tabular.add_row(
            "Observations",
            *[int(nobs["5 years"])] * 2,
            *[int(nobs["10 years"])] * 3,
        )

    elif version == 2:
        tabular.add_row(
            "Observations",
            *[int(nobs["10 years"])] * 3,
            *[int(nobs["3 years"])] * 2,
            *[int(nobs["5 years"])] * 2,
        )

    elif version == 3:
        tabular.add_row(
            "Observations",
            *[int(nobs["10 years"])] * 3,
            *[int(nobs["3 years"])] * 2,
            *[int(nobs["5 years"])] * 3,
        )
    elif version == 4:
        tabular.add_row(
            "Observations",
            *[int(nobs["5 years"])] * 2,
            *[int(nobs["3 years"])] * 2,
            *[int(nobs["10 years"])] * 3,
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


def _format_coefs_and_ses(coefs, ses):
    """Add significance stars to coefficients.

    Args:
        coefs (dict): Dictionary with coefficients.
        ses (dict): Dictionary with standard errors.

    Returns:
        adj_coefs (dict): Coefficients with significance stars.
        adj_ses (dict): Standard errors with brackets.

    """
    adj_coefs = {key: None for key in coefs}
    adj_ses = {key: None for key in ses}

    for key in coefs:
        adj_coefs[key] = f"{coefs[key]:.2f}" + _stars_based_on_t_stat(
            coefs[key],
            ses[key],
        )
        adj_ses[key] = "(" + f"{ses[key]:.2f}" + ")"

    return adj_coefs, adj_ses


def _stars_based_on_t_stat(coef, se):
    if abs(coef / se) > 2.58:
        return "***"

    elif abs(coef / se) > 1.96:
        return "**"

    elif abs(coef / se) > 1.64:
        return "*"

    else:
        return ""


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
