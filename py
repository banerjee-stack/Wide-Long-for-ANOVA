"""
wide_to_long_anova.py

Utility function to reshape wide → long data and run
repeated-measures or mixed ANOVA using pingouin.

This mirrors the functionality found in the R version
(run_wide_to_long_anova) used in your repositories.

Dependencies:
    pandas
    pingouin
"""

from __future__ import annotations
import pandas as pd
import re
import pingouin as pg
from typing import List, Optional, Union, Dict


def run_wide_to_long_anova(
    data: pd.DataFrame,
    id_col: str,
    dv_name: str,
    within_cols: Union[List[str], str],
    within_name: str = "condition",
    between_col: Optional[str] = None,
    posthoc_correction: str = "bonf"
) -> Dict[str, pd.DataFrame]:
    """
    Convert wide-format repeated measures into long format and run ANOVA.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset in wide format.
    id_col : str
        Column identifying subjects (e.g., "PARTICIPANT").
    dv_name : str
        Name for dependent variable column in long data.
    within_cols : list or str
        Repeated-measures columns (list of names OR "regex:^pattern").
    within_name : str, default="condition"
        Name of long-format repeated-measures factor.
    between_col : str or None, default=None
        Optional between-subjects factor column.
    posthoc_correction : str, default="bonf"
        Correction for multiple comparisons in post-hocs.

    Returns
    -------
    dict
        {
            "long_data": long-format dataframe,
            "anova_table": ANOVA results from pingouin,
            "posthoc_table": post-hoc test results
        }

    Notes
    -----
    - Uses `pingouin.rm_anova` for repeated measures.
    - Uses `pingouin.mixed_anova` for mixed designs.
    - Post-hocs performed via `pingouin.pairwise_ttests`.
    """

    df = data.copy()

    # -----------------------------------------
    # Handle regex for within-subject variables
    # -----------------------------------------
    if isinstance(within_cols, str) and within_cols.startswith("regex:"):
        pattern = within_cols.replace("regex:", "")
        matched = [c for c in df.columns if re.search(pattern, c)]

        if not matched:
            raise ValueError(
                f"No columns matched regex pattern '{pattern}'. "
                f"Available columns: {list(df.columns)}"
            )
        within_cols = matched

    # Columns needed for analysis
    keep = [id_col] + within_cols
    if between_col:
        keep.append(between_col)

    df = df[keep]

    # -----------------------------------------
    # Wide → Long reshaping
    # -----------------------------------------
    long_df = df.melt(
        id_vars=[id_col] + ([between_col] if between_col else []),
        value_vars=within_cols,
        var_name=within_name,
        value_name=dv_name
    )

    # -----------------------------------------
    # Run ANOVA
    # -----------------------------------------
    if between_col:
        # Mixed ANOVA
        anova_res = pg.mixed_anova(
            data=long_df,
            dv=dv_name,
            within=within_name,
            between=between_col,
            subject=id_col
        )
    else:
        # Repeated-measures ANOVA
        anova_res = pg.rm_anova(
            data=long_df,
            dv=dv_name,
            within=within_name,
            subject=id_col,
            detailed=True
        )

    # -----------------------------------------
    # Post-hoc pairwise tests
    # -----------------------------------------
    posthoc = pg.pairwise_ttests(
        data=long_df,
        dv=dv_name,
        within=within_name,
        between=between_col,
        subject=id_col,
        padjust=posthoc_correction,
        effsize="hedges"
    )

    return {
        "long_data": long_df,
        "anova_table": anova_res,
        "posthoc_table": posthoc
    }
