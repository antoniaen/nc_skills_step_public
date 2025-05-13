"""Plausible Values Method."""


def plausible_values_method(coefs, sampl_vars, model_names, n):
    """Combine results from each of 10 plausible values to get overall estimates.

    Args:
        coefs (dict): Coefficients from each plausible value. Keys are model_names
        sampl_vars (dict): Sample variance from each plausible value. Keys are model_names.
        model_names (list): Names of models.
        n (int): Number of plausible values.

    Returns:
        final_coefs (dict): Final coefficients.
        final_std_errors (dict): Final standard errors.

    """
    final_coefs = {model: None for model in model_names}
    final_sampl_vars = {model: None for model in model_names}
    imputation_vars = {model: None for model in model_names}
    final_std_errors = {model: None for model in model_names}

    for model in model_names:
        final_coefs[model] = sum(coefs[model]) / len(coefs[model])
        final_sampl_vars[model] = sum(sampl_vars[model]) / len(sampl_vars[model])

        imputation_vars[model] = (
            sum((coefs[model][i] - final_coefs[model]) ** 2 for i in range(n)) / n
        )

        # Final sample variance and imputation variance have to be combined to get the
        # total variance.
        final_std_errors[model] = (
            final_sampl_vars[model] + (1 + (1 / n)) * imputation_vars[model]
        ) ** (1 / 2)

    return final_coefs, final_std_errors
