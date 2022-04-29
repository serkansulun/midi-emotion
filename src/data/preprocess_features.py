import pandas as pd
import numpy as np

def preprocess_features(feature_file, n_bins=None, min_n_instruments=3, 
        test_ratio=0.05, outlier_range=1.5, conditional=True,
        use_labeled_only=True):

    # Preprocess data
    data = pd.read_csv(feature_file)
    mapper = {"valence": "valence", "note_density_per_instrument": "arousal"}
    data = data.rename(columns=mapper)
    columns = data.columns.to_list()

    # filter out ones with less instruments
    data = data[data["n_instruments"] >= min_n_instruments]
    # filter out ones with zero valence
    data = data[data["valence"] != 0]

    # filter out outliers
    feature_labels = list(mapper.values())
    outlier_indices = []
    for label in feature_labels:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + outlier_range * iqr
        lower_limit = q1 - outlier_range * iqr

        outlier_indices += series[series < lower_limit].index.to_list()
        outlier_indices += series[series > upper_limit].index.to_list()
    data.drop(outlier_indices, inplace=True)

    # shift and scale features between -1 and 1
    for label in feature_labels:
        series = data[label]
        min_ = series.min()
        max_ = series.max()
        
        data[label] = (data[label] - min_) / (max_ - min_) * 2 - 1

    if n_bins is not None:
        # digitize into bins using quantiles
        quantile_indices = np.linspace(0, 1, n_bins+1)
        for label in feature_labels:

            # create token labels
            if n_bins % 2 == 0:
                bin_ids = list(range(-n_bins//2, 0)) + list(range(1, n_bins//2+1))
            else:
                bin_ids = list(range(-(n_bins-1)//2, (n_bins-1)//2 + 1))
            token_labels = ["<{}{}>".format(label[0].upper(), bin_id) \
                for bin_id in bin_ids]
            # additional label for NaN (missing) values: <V>
            token_labels.append(None)   # to handle NaNs    
            
            series = data[label]
            quantiles = [series.quantile(q) for q in quantile_indices]
            quantiles[-1] += 1e-6
            series = series.to_numpy()
            series_digitized = np.digitize(series, quantiles)
            series_tokenized = [token_labels[i-1] for i in series_digitized]

            data[label] = series_tokenized
    else:
        # convert NaN into None
        data = data.where(pd.notnull(data), None)

    # Create train and test splits
    matched = data[data["is_matched"]]
    unmatched = data[~data["is_matched"]]

    # reserve a portion of matched data for testing
    matched = matched.sort_values("file")
    matched = matched.reset_index(drop=True)
    n_test_samples = round(len(matched) * test_ratio)

    test_split = matched.loc[len(matched)-n_test_samples:len(matched)]

    train_split = matched.loc[:len(matched)-n_test_samples]

    if not use_labeled_only:
        train_split = pd.concat([train_split, unmatched])
        train_split = train_split.sort_values("file").reset_index(drop=True)

    splits = [train_split, test_split]

    # summarize
    columns_to_drop = [col for col in columns if col not in ["file", "valence", "arousal"]]
    if not conditional:
        columns_to_drop += ["valence", "arousal"]

    # filter data so all features are valid (not None = matched data)
    for label in feature_labels:
        # test split has to be identical across vanilla and conditional models
        splits[1] = splits[1][~splits[1][label].isnull()]

        # filter train split only for conditional models
        if use_labeled_only:
            splits[0] = splits[0][~splits[0][label].isnull()]

    for i in range(len(splits)):
        # summarize
        splits[i] = splits[i].drop(columns=columns_to_drop, errors="ignore")
        splits[i] = splits[i].to_dict("records")    

    return splits