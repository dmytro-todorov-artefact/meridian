import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az

import IPython

from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter

from on_top_functionality.paths import input_params_path, campaign_df_path, input_df_path, collected_data_path, mc_df_chan_path, summary_metrics_chan_path, predictive_accuracy_chan_path, mc_df_camp_path, summary_metrics_camp_path, predictive_accuracy_camp_path
from on_top_functionality.data_processing.config_generation import get_params_from_excel
from on_top_functionality.constants import (
    DATE,
    CTRB_PREFIX,
    REVENUE_PER_KPI,
    IMPR_PREFIX,
    SPEND_PREFIX,
    START_IMPR_PREFIX,
    START_SPEND_PREFIX,
    TARGET,
)
from on_top_functionality.parameters import sample_prior, n_chains, n_adapt, n_burnin, n_keep


class _BaseModel():
  def __init__(self, retailer: str, controls: list = [], roi_start_date: str = None, roi_end_date: str = None, knots_frequency: float = None):
    self.retailer = retailer
    self.controls = controls
    self.roi_start_date = roi_start_date
    self.roi_end_date = roi_end_date
    self.target = TARGET
    self.level = "channel"
    self.csv_path = input_df_path.format(self.retailer)
    self.knots_frequency = knots_frequency


  def read_data_from_excel(self, sheet_name):
    df = pd.read_excel(collected_data_path.format(self.retailer), sheet_name=sheet_name)
    df = df.drop(df.columns[0], axis=1).rename(columns={'Unnamed: 1': DATE})
    df[DATE] = pd.to_datetime(df[DATE])
    df.set_index(DATE, inplace=True)
    df.rename(columns={col: col.replace(START_IMPR_PREFIX, IMPR_PREFIX).replace(START_SPEND_PREFIX, SPEND_PREFIX) for col in df.columns}, inplace=True)
    return df

  def preprocessing(self):
    pass


  def load_data(self, target, csv_path):
    self.impr_columns = [col for col in self.df.columns if col.startswith(IMPR_PREFIX)]
    self.spend_columns = [col for col in self.df.columns if col.startswith(SPEND_PREFIX)]
    self.media_columns = [col.replace(IMPR_PREFIX, '') for col in self.impr_columns]

    self.coord_to_columns = load.CoordToColumns(
        time=DATE,
        # geo='geo',
        controls=self.controls,
        # population='population',
        kpi=target,
        revenue_per_kpi=REVENUE_PER_KPI,
        media=self.impr_columns,
        media_spend=self.spend_columns,
    )

    self.correct_media_to_channel = {col: col.replace(IMPR_PREFIX, '') for col in self.impr_columns}
    self.correct_media_spend_to_channel = {col: col.replace(SPEND_PREFIX, '') for col in self.spend_columns}
    loader = load.CsvDataLoader(
        csv_path=csv_path,
        kpi_type='non_revenue',
        coord_to_columns=self.coord_to_columns,
        media_to_channel=self.correct_media_to_channel,
        media_spend_to_channel=self.correct_media_spend_to_channel,
    )
    self.data = loader.load()


  def load_priors(self, level):
    params = get_params_from_excel(input_params_path.format(self.retailer))
    if level == 'channel':
      priors = params['ROI_INTERVALS']
    elif level == 'campaign':
      priors = params['ROIS_SEGMENT'][self.channel]

    build_media_channel_args = self.data.get_paid_media_channels_argument_builder()

    roi_m = build_media_channel_args(**priors) # This creates a list of channel-ordered (mu, sigma) tuples.
    roi_m_low, roi_m_high = zip(*roi_m)

    prior = prior_distribution.PriorDistribution(
      roi_m=tfp.distributions.Uniform(
        roi_m_low, roi_m_high, name=constants.ROI_M
        )
    )

    self.prior = prior

  def set_roi_calibration_period(self):
    roi_dates = [date for date in self.data.time.values if self.roi_start_date <= date <= self.roi_end_date]
    roi_period = {
      channel: roi_dates if self.data.media.loc[:,roi_dates,channel].sum()!=0 else self.data.time.values.tolist() for channel in self.data.media_channel.values
    }

    roi_calibration_period = np.zeros((len(self.data.time), len(self.data.media_channel)))
    for i in roi_period.items():
      roi_calibration_period[
          np.isin(self.data.time.values, i[1]), self.data.media_channel.values == i[0]
      ] = 1

    roi_calibration_period[
        :, ~np.isin(self.data.media_channel.values, list(roi_period.keys()))
    ] = 1

    self.roi_calibration_period = roi_calibration_period


  def set_knots(self):
    self.knots = round(self.knots_frequency * len(self.data.time.values))

  def modeling(self, **kwargs):
    model_spec = spec.ModelSpec(prior=self.prior, roi_calibration_period=self.roi_calibration_period, **kwargs)
    self.model = model.Meridian(input_data=self.data, model_spec=model_spec)
    self.model.sample_prior(sample_prior)
    self.model.sample_posterior(n_chains=n_chains, n_adapt=n_adapt, n_burnin=n_burnin, n_keep=n_keep)

  def postprocessing(self):
    self.analyzer = analyzer.Analyzer(self.model)
    time_summary_metrics = self.analyzer.summary_metrics(use_kpi=True, aggregate_times=False).to_dataframe().reset_index()
    time_summary_metrics = time_summary_metrics[(time_summary_metrics['distribution'] != 'posterior') & (time_summary_metrics['metric'] == 'mean')].drop(columns=['distribution', 'metric'])
    self.mc_df = time_summary_metrics.pivot(index='time', columns='channel', values='incremental_outcome')
    self.mc_df.to_csv(self.mc_df_path)

    summary_metrics = self.analyzer.summary_metrics(use_kpi=True, aggregate_times=False).to_dataframe().reset_index()
    self.summary_metrics = summary_metrics[(summary_metrics['distribution'] != 'posterior') & (summary_metrics['metric'] == 'mean')].drop(columns=['distribution', 'metric'])
    self.summary_metrics.to_csv(self.summary_metrics_path, index=False)

    self.predictive_accuracy = self.analyzer.predictive_accuracy().to_dataframe().reset_index()
    self.predictive_accuracy.to_csv(self.predictive_accuracy_path, index=False)



  def run(self):
    print("Preprocessing...")
    self.preprocessing()
    print("Loading data...")
    self.load_data(self.target, self.csv_path)
    print("Loading priors...")
    self.load_priors(self.level)
    print("Setting ROI calibration period...")
    self.set_roi_calibration_period()
    print("Modeling...")
    kwargs = {}
    if self.knots_frequency:
      self.set_knot
      kwargs['knots'] = self.knots

    self.modeling(**kwargs)
    print("Postprocessing...")
    self.postprocessing()
    return self


class ChannelModel(_BaseModel):
  def __init__(self, retailer: str, controls: list = [], roi_start_date: str = None, roi_end_date: str = None):
    super().__init__(retailer=retailer, controls=controls, roi_start_date=roi_start_date, roi_end_date=roi_end_date)
    self.target = TARGET
    self.level = 'channel'
    self.csv_path = input_df_path.format(self.retailer)
    self.mc_df_path = mc_df_chan_path.format(self.retailer)
    self.summary_metrics_path = summary_metrics_chan_path.format(self.retailer)
    self.predictive_accuracy_path = predictive_accuracy_chan_path.format(self.retailer)


  def preprocessing(self):
    df_ctrl = self.read_data_from_excel(f"{self.retailer} non media drivers")
    df_chan = self.read_data_from_excel(f"{self.retailer} channel level")
    df = pd.concat([df_ctrl, df_chan], axis=1)
    df.to_csv(input_df_path.format(self.retailer))
    self.df = df


class CampaignModel(_BaseModel):
  def __init__(self, retailer: str, channel, controls: list = [], roi_start_date: str = None, roi_end_date: str = None):
    super().__init__(retailer=retailer, controls=controls, roi_start_date=roi_start_date, roi_end_date=roi_end_date)
    self.channel = channel
    self.target = f'{CTRB_PREFIX}{channel}'
    self.level = 'campaign'
    self.csv_path = campaign_df_path.format(self.retailer, self.channel)
    self.mc_df_path = mc_df_camp_path.format(self.retailer, self.channel)
    self.summary_metrics_path = summary_metrics_camp_path.format(self.retailer, self.channel)
    self.predictive_accuracy_path = predictive_accuracy_camp_path.format(self.retailer, self.channel)


  def preprocessing(self):
    df_camp = self.read_data_from_excel(f"{self.retailer} campaign level - {self.channel}")
    mc_df = pd.read_csv(mc_df_chan_path.format(self.retailer))
    df = pd.read_csv(input_df_path.format(self.retailer))
    df_camp[f'{CTRB_PREFIX}{self.channel}'] = mc_df[self.channel].values
    df_camp[REVENUE_PER_KPI] = df[REVENUE_PER_KPI].values
    df_camp.to_csv(campaign_df_path.format(self.retailer, self.channel))
    self.df = df_camp
