import logging
import os

import pandas as pd

from selection import feature_selection
from selection.log_decorator import LogDecorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@LogDecorator('Correlation')
def select(df: pd.DataFrame, num_features=100, force=True):
  feature_fn = 'features/Correlation.csv'
  if not os.path.exists(feature_fn) or force:
    os.makedirs('features', exist_ok=True)
    _, corr_features = feature_selection.correlation_with_class(df, num_features)
    pd.DataFrame(corr_features).to_csv(feature_fn)
    X = df[corr_features].T.values  # input size x batch size
  else:
    fisher_features = pd.read_csv(feature_fn, index_col=1)
    X = df[fisher_features.index].T.values  # input size x batch size
  return X
