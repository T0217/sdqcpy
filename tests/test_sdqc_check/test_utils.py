import pandas as pd
import numpy as np
from sdqc_check.utils import combine_data_and_labels, set_seed


def test_combine_data_and_labels(raw_data, synthetic_data):
    X, y = combine_data_and_labels(raw_data, synthetic_data)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(raw_data) + len(synthetic_data)
    assert isinstance(y, np.ndarray)
    assert len(y) == len(X)
    assert np.sum(y == 1) == len(raw_data)
    assert np.sum(y == 0) == len(synthetic_data)


def test_set_seed():
    seed = 17
    set_seed(seed)

    random_1 = np.random.rand()
    set_seed(seed)
    random_2 = np.random.rand()
    assert random_1 == random_2

    import os
    assert os.environ['PYTHONHASHSEED'] == str(seed)

    import torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        random_tensor_1 = torch.rand(1)
        set_seed(seed)
        random_tensor_2 = torch.rand(1)
        assert torch.all(random_tensor_1.eq(random_tensor_2))
