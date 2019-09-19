import numpy as np
import pandas as pd


def fisher(data_df: pd.DataFrame, num_features):
  """
  Makes feature Fisher selection according to the following formula:
  D(f) = m1(f) - m2(f) / sigma1(f) + sigma2(f)
  :return: the list of most significant features.
  """
  fisher_df = pd.DataFrame()
  fisher_df['mean_autism'] = data_df[data_df['Case'] == 'AUTISM'].mean()
  fisher_df['mean_control'] = data_df[data_df['Case'] == 'CONTROL'].mean()
  fisher_df['std_autism'] = data_df[data_df['Case'] == 'AUTISM'].std()
  fisher_df['std_control'] = data_df[data_df['Case'] == 'CONTROL'].std()
  
  fisher_df['diff_mean'] = (fisher_df['mean_autism'] - fisher_df['mean_control']).abs()
  fisher_df['sum_std'] = (fisher_df['std_autism'] + fisher_df['std_autism']).abs()
  fisher_df['fisher_coeff'] = fisher_df['diff_mean'] / fisher_df['sum_std']
  fisher_df = fisher_df.sort_values(['fisher_coeff'], ascending=False)
  
  # data_df['fisher_coeff'] = fisher_df['fisher_coeff']
  most_significant_features = fisher_df['fisher_coeff'].index[:num_features]
  return data_df, most_significant_features


def correlation_with_class(data_df: pd.DataFrame, num_features):
  """
  Makes feature correlation with class selection according to the following formula:
  D(f) = [(m1(f) - m(f))^2 + (m2(f) - m(f))^2] / 2*sigma(f)^2
  :return: the list of most significant features.
  """
  corr_df = pd.DataFrame()
  corr_df['mean_autism'] = data_df[data_df['Case'] == 'AUTISM'].mean()
  corr_df['mean_control'] = data_df[data_df['Case'] == 'CONTROL'].mean()
  corr_df['mean_total'] = data_df.mean()
  corr_df['std_total'] = data_df.std()
  
  corr_df['corr_coeff'] = ((corr_df['mean_autism'] - corr_df['mean_total']) ** 2 + (
      corr_df['mean_control'] - corr_df['mean_total']) ** 2) / 2 * (corr_df['std_total']) ** 2
  corr_df = corr_df.sort_values(['corr_coeff'], ascending=False)
  most_significant_features = corr_df['corr_coeff'].index[:num_features]
  # data_df['corr_coeff'] = corr_df['corr_coeff']
  return data_df, most_significant_features


def t_test(data_df: pd.DataFrame, num_features):
  """
  :return: the list of most significant features.
  """
  t_test = pd.DataFrame()
  t_test['mean_autism'] = data_df[data_df['Case'] == 'AUTISM'].mean()
  t_test['mean_control'] = data_df[data_df['Case'] == 'CONTROL'].mean()
  t_test['std_autism'] = data_df[data_df['Case'] == 'AUTISM'].std()
  t_test['std_control'] = data_df[data_df['Case'] == 'CONTROL'].std()
  
  num_autism = len(data_df[data_df['Case'] == 'AUTISM'])
  num_control = len(data_df[data_df['Case'] == 'CONTROL'])
  
  t_test['ttest_coeff'] = (t_test['mean_autism'] - t_test['mean_control']) / (
      (t_test['std_autism'] ** 2) / num_autism + (t_test['std_control'] ** 2) / num_control) ** (
                              1 / 2)  # TODO is it correct?
  t_test = t_test.sort_values(['ttest_coeff'], ascending=False)
  most_significant_features = t_test['ttest_coeff'].index[:num_features]
  # data_df['ttest_coeff'] = t_test['ttest_coeff']
  return data_df, most_significant_features


def random(data_df: pd.DataFrame, num_features):  # TODO
  """
  Random features selection for baseline purposes.
  :return: the list of randomly chosen features.
  """
  total_features = data_df.shape[1]
  features = np.arange(total_features)
  np.random.shuffle(features)
  selected_random = features[:num_features]
  most_significant_features = data_df.T.index[selected_random]
  return data_df, most_significant_features
