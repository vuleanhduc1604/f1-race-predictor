import sys
sys.path.insert(0, '.')
import fastf1
import pandas as pd
from src.config import CACHE_DIR, MODELS_DIR
from src.data.loaders import build_training_dataset
from src.utils.helpers import get_drop_columns

fastf1.Cache.enable_cache(str(CACHE_DIR))
fastf1.Cache.offline_mode = True

_, test_data = build_training_dataset(test_year=2025)

from src.models.ranker import RaceRanker
ranker = RaceRanker.load(MODELS_DIR / 'ranker.pkl')

drop_cols = get_drop_columns(test_data)
X = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
test_data['predicted_position'] = ranker.predict_positions(X, test_data['EventName'])
test_data['_err'] = (test_data['predicted_position'] - test_data['Position']).abs()

top7 = test_data[test_data['Position'] <= 7]
report = (
    top7.groupby('EventName')
    .agg(mae=('_err', 'mean'))
    .sort_values('mae')
)
print(report.to_string())