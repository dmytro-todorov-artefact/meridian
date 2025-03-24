import os
from typing import Optional

import numpy as np
import pandas as pd

from on_top_functionality.data_processing.config_column_names import (
    CHANNELS_FOR_REDISTRIBUTION_VAR,
    CHANNELS_WITH_PRIORS_VAR,
    CHANNELS_WITHOUT_PRIORS_VAR,
    CONTROL_CONSTRAINTS_VAR,
    GAMMA_ADJUSTMENT_VAR,
    GROSS_MARGIN_VAR,
    HALF_LIFE_ADJUSTMENT_VAR,
    HALF_LIFE_CAMPAIGN_ADJUSTMENT_VAR,
    HALF_LIFE_VALUES_VAR,
    HALO_CHANNELS_WITH_PRIORS_VAR,
    HALO_DIRECT_PRIORS_ORIGINAL_VAR,
    HALO_DIRECT_PRIORS_VAR,
    REQUESTED_SEGMENTS_VAR,
    ROI_INTERVALS_VAR,
    ROI_RANGE_VAR,
    ROIS_SEGMENT_VAR,
    SATURATION_ADJUSTMENT_VAR,
)
from on_top_functionality.data_processing.dataframe_columns import WE
from on_top_functionality.data_processing.excel_column_names import (
    CAMPAIGN_COLUMN,
    CHANNEL_COLUMN,
    HALO_DIRECT_COLUMN,
    LB_NR_ROI_COLUMN,
    NR_ROI_COLUMN,
    SEGMENT_COLUMN,
    UP_NR_ROI_COLUMN,
)


def return_halo_dict(df):
    df = df.set_index(CAMPAIGN_COLUMN)[[HALO_DIRECT_COLUMN, SEGMENT_COLUMN]].T.to_dict()
    df = {key: list(dict_value.values()) for key, dict_value in df.items()}
    return df


def return_dict_with_bounds(group):
    lower = group.set_index(CAMPAIGN_COLUMN).to_dict()[LB_NR_ROI_COLUMN]
    upper = group.set_index(CAMPAIGN_COLUMN).to_dict()[UP_NR_ROI_COLUMN]
    result = {
        key: [lower[key], upper[key]] for key in lower.keys()
    }  # if not np.isnan(lower[key])}
    return result


def return_rois_segment_dict(df):
    rois_segment = {
        channel: return_dict_with_bounds(group)
        for channel, group in df.groupby(CHANNEL_COLUMN)
    }

    return rois_segment


def get_params_from_excel(input_params_path, campaign_values: str = "list", STD_SCALE=False):
    """
    Reads Input params excel file and generates config file

    Args:
        COUNTRY (str): COUNTRY
        BRAND (str): BRAND
        GROSS_MARGIN (float, optional): GROSS_MARGIN. Defaults to 0.
        campaign_values (str, optional): "list" or "float". Defaults to "list". Set campaign params as bounds [LB NR ROI, UP NR ROI] if "list" or as NR ROI if "float"
        STD_SCALE (bool, optional): standardize control features std. Defaults to False.

    Returns:
        dict: dictionary with config parameters
    """
    input_channel = pd.read_excel(
        input_params_path, sheet_name=CHANNEL_COLUMN, index_col=CHANNEL_COLUMN
    )
    input_campaign = pd.read_excel(input_params_path, sheet_name=CAMPAIGN_COLUMN)
    input_segment = pd.read_excel(input_params_path, sheet_name=SEGMENT_COLUMN)
    input_gm = pd.read_excel(
        input_params_path, sheet_name="Gross Margin", index_col="Year"
    )

    input_campaign_chan_priors = input_campaign.merge(
        input_channel, on=CHANNEL_COLUMN, how="left"
    )
    input_campaign = input_campaign.merge(
        input_segment, on=[CHANNEL_COLUMN, SEGMENT_COLUMN], how="left"
    )

    for col in [NR_ROI_COLUMN, LB_NR_ROI_COLUMN, UP_NR_ROI_COLUMN]:
        input_campaign[col] = input_campaign[col].fillna(input_campaign_chan_priors[col])
    params = {}

    params[CONTROL_CONSTRAINTS_VAR] = {}
    params[CHANNELS_FOR_REDISTRIBUTION_VAR] = input_channel.index.to_list()
    params[CHANNELS_WITH_PRIORS_VAR] = input_channel.index[
        input_channel[[NR_ROI_COLUMN, LB_NR_ROI_COLUMN, UP_NR_ROI_COLUMN]]
        .notna()
        .all(axis=1)
    ].to_list()
    params[CHANNELS_WITHOUT_PRIORS_VAR] = [
        channel
        for channel in params[CHANNELS_FOR_REDISTRIBUTION_VAR]
        if channel not in params[CHANNELS_WITH_PRIORS_VAR]
    ]
    params[HALO_CHANNELS_WITH_PRIORS_VAR] = params[
        CHANNELS_FOR_REDISTRIBUTION_VAR
    ].copy()  # params[CHANNELS_WITH_PRIORS].copy()

    half_life_values = input_channel[input_channel["Half life"].notna()].to_dict()[
        "Half life"
    ]

    # def half_life_adj_value(value):
    #     if value <= 0.8:
    #         return [8,3]
    #     elif value <= 1.2:
    #         return [15,3]
    #     elif value <= 2.2:
    #         return [40,3]
    #     else:
    #         return [40,1.5]

    # params[HALF_LIFE_ADJUSTMENT] = {channel: half_life_adj_value(value) for channel, value in half_life_values.items()}
    # params[HALF_LIFE_CAMPAIGN_ADJUSTMENT] = {}
    params[SATURATION_ADJUSTMENT_VAR] = {}
    params[GAMMA_ADJUSTMENT_VAR] = {}
    params[GROSS_MARGIN_VAR] = input_gm.to_dict()["Value"]
    params[HALF_LIFE_VALUES_VAR] = half_life_values

    ROI_INTERVALS = input_channel[[LB_NR_ROI_COLUMN, UP_NR_ROI_COLUMN]].T
    ROI_INTERVALS = ROI_INTERVALS[
        ROI_INTERVALS.columns[~ROI_INTERVALS.isna().all()]
    ].to_dict()
    params[ROI_INTERVALS_VAR] = {
        key: tuple(dict_value.values()) for key, dict_value in ROI_INTERVALS.items()
    }

    if campaign_values == "list":

        def return_dict_with_bounds(group):
            lower = group.set_index(CAMPAIGN_COLUMN).to_dict()[LB_NR_ROI_COLUMN]
            upper = group.set_index(CAMPAIGN_COLUMN).to_dict()[UP_NR_ROI_COLUMN]
            result = {
                key: (lower[key], upper[key]) for key in lower.keys()
            }  # if not np.isnan(lower[key])}
            return result

        params[ROIS_SEGMENT_VAR] = {
            channel: return_dict_with_bounds(group)
            for channel, group in input_campaign.groupby(CHANNEL_COLUMN)
        }
    else:
        params[ROIS_SEGMENT_VAR] = {
            channel: group.set_index(CAMPAIGN_COLUMN).to_dict()[NR_ROI_COLUMN]
            for channel, group in input_campaign.groupby(CHANNEL_COLUMN)
        }

    params[REQUESTED_SEGMENTS_VAR] = input_campaign[CAMPAIGN_COLUMN].to_list()
    input_campaign_for_halo = input_campaign.copy()
    input_campaign_for_halo[HALO_DIRECT_COLUMN] = input_campaign_for_halo[HALO_DIRECT_COLUMN].fillna(1)
    input_campaign_for_halo[SEGMENT_COLUMN] = input_campaign_for_halo[SEGMENT_COLUMN].fillna("BRAND")
    HALO_DIRECT_PRIORS_original = (
        input_campaign_for_halo.groupby([CHANNEL_COLUMN, SEGMENT_COLUMN])[
            HALO_DIRECT_COLUMN
        ]
        .first()
        .reset_index()
    )
    params[HALO_DIRECT_PRIORS_ORIGINAL_VAR] = {
        channel: group.set_index(SEGMENT_COLUMN).to_dict()[HALO_DIRECT_COLUMN]
        for channel, group in HALO_DIRECT_PRIORS_original.groupby(CHANNEL_COLUMN)
    }

    params[HALO_DIRECT_PRIORS_VAR] = {
        channel: return_halo_dict(group)
        for channel, group in input_campaign_for_halo.groupby(CHANNEL_COLUMN)
    }

    return params
