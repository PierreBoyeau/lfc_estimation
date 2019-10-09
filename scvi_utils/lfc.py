from scvi.inference import UnsupervisedTrainer
from scvi.utils import demultiply, compute_hdi, softmax
import os
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import warnings
import pickle


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


def train_model(
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    filename: str = None,
):
    """

    :param mdl_class: Class of algorithm
    :param dataset: Dataset
    :param mdl_params:
    :param train_params:
    :param train_fn_params:
    :param filename
    :return:
    """
    # if os.path.exists(filename):
    #     res = load_pickle(filename)
    #     return res["vae"], res["trainer"]

    if "test_indices" not in train_params:
        warnings.warn("No `test_indices` attribute found.")
    my_vae = mdl_class(dataset.nb_genes, n_batch=dataset.n_batches, **mdl_params)
    my_trainer = UnsupervisedTrainer(my_vae, dataset, **train_params)
    my_trainer.train(**train_fn_params)
    print(my_trainer.train_losses)
    return my_vae, my_trainer


def estimate_lfc_density(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    n_picks: int = 10,
    n_samples: int = 500,
    label_a=0,
    label_b=1,
    importance_sampling=False
):
    """

    """

    lfcs = dict()
    my_vae, my_trainer = train_model(
        mdl_class, dataset, mdl_params, train_params, train_fn_params
    )
    post = my_trainer.test_set
    train_indices = post.data_loader.sampler.indices
    train_samples = np.random.permutation(train_indices)
    post = my_trainer.create_posterior(
        model=my_vae, gene_dataset=dataset, indices=train_samples
    )
    outputs = post.get_latents(n_samples=n_samples, other=True, device="cpu")
    z, labels, scales = outputs["z"], outputs["label"], outputs["scale"]
    weights = softmax(outputs["log_probas"], axis=0)

    for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
        lfc_size = []
        for exp in range(n_picks):
            labels = labels.squeeze()
            # Sampling cells
            where_a = np.where(labels == label_a)[0]
            where_b = np.where(labels == label_b)[0]
            where_a = where_a[np.random.choice(len(where_a), size=size)]
            where_b = where_b[np.random.choice(len(where_b), size=size)]
            # Sampling z observations
            scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
            scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
            if importance_sampling:
                weights_a = weights[:, where_a].reshape((-1)).numpy() / len(where_a)
                weights_b = weights[:, where_b].reshape((-1)).numpy() / len(where_b)
            else:
                weights_a = None
                weights_b = None
            scales_a, scales_b = demultiply(
                arr1=scales_a,
                arr2=scales_b,
                factor=3,
                weights_a=weights_a,
                weights_b=weights_b
            )
            lfc = np.log2(scales_a) - np.log2(scales_b)
            assert not np.isnan(lfc).any(), lfc
            lfc_size.append(lfc)
        lfc_size = np.array(lfc_size)
        lfcs[size] = lfc_size
    save_pickle(lfcs, filename=filename)
    return lfcs


def estimate_lfc_mean(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    n_picks: int = 10,
    n_samples: int = 500,
    label_a=0,
    label_b=1,
    importance_sampling=False
) -> dict:
    """
        Returns LFC POINT ESTIMATES
    """
    if os.path.exists(filename):
        return load_pickle(filename)
    lfcs = dict()
    my_vae, my_trainer = train_model(
        mdl_class, dataset, mdl_params, train_params, train_fn_params
    )
    post = my_trainer.test_set
    train_indices = post.data_loader.sampler.indices
    train_samples = np.random.permutation(train_indices)
    post = my_trainer.create_posterior(
        model=my_vae, gene_dataset=dataset, indices=train_samples
    )
    outputs = post.get_latents(n_samples=n_samples, other=True, device="cpu")
    z, labels, scales = outputs["z"], outputs["label"], outputs["scale"]
    weights = softmax(outputs["log_probas"], axis=0)

    for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
        lfc_size = []
        for exp in range(n_picks):
            labels = labels.squeeze()
            # Sampling cells
            where_a = np.where(labels == label_a)[0]
            where_b = np.where(labels == label_b)[0]
            where_a = where_a[np.random.choice(len(where_a), size=size)]
            where_b = where_b[np.random.choice(len(where_b), size=size)]
            # Sampling z observations
            scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
            scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
            if importance_sampling:
                weights_a = weights[:, where_a].reshape((-1)).numpy() / len(where_a)
                weights_b = weights[:, where_b].reshape((-1)).numpy() / len(where_b)
            else:
                weights_a = None
                weights_b = None
            scales_a, scales_b = demultiply(
                arr1=scales_a,
                arr2=scales_b,
                factor=3,
                weights_a=weights_a,
                weights_b=weights_b
            )
            lfc = np.log2(scales_a) - np.log2(scales_b)
            # assert not np.isnan(lfc).any(), lfc
            if np.isnan(lfc).any():
                warnings.warn("NaN values appeared in LFCs")
            lfc_size.append(lfc.mean(0))
        lfc_size = np.array(lfc_size)
        lfcs[size] = lfc_size
    save_pickle(lfcs, filename=filename)
    return lfcs


def estimate_de_proba(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    delta: float = 0.5,
    n_trainings: int = 5,
    n_picks: int = 25,
    n_samples: int = 500,
    label_a=0,
    label_b=1,
):
    """

    """
    if os.path.exists(filename):
        return np.load(filename)

    n_sizes = len(sizes)
    de_probas = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes))
    #     lfcs = np.zeros((n_trainings, N_SIZES, n_picks, dataset.nb_genes, 3*n_samples))
    for training in range(n_trainings):
        my_vae, my_trainer = train_model(
            mdl_class, dataset, mdl_params, train_params, train_fn_params
        )
        post = my_trainer.test_set
        train_indices = post.data_loader.sampler.indices
        train_samples = np.random.permutation(train_indices)
        post = my_trainer.create_posterior(
            model=my_vae, gene_dataset=dataset, indices=train_samples
        )
        outputs = post.get_latents(
            n_samples=n_samples, other=True, device="cpu"
        )
        z, labels, scales = outputs["z"], outputs["label"], outputs["scale"]

        for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
            for exp in range(n_picks):
                labels = labels.squeeze()
                where_a = np.where(labels == label_a)[0]
                where_b = np.where(labels == label_b)[0]
                where_a = where_a[np.random.choice(len(where_a), size=size)]
                where_b = where_b[np.random.choice(len(where_b), size=size)]
                scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
                scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
                scales_a, scales_b = demultiply(arr1=scales_a, arr2=scales_b, factor=3)
                lfc = np.log2(scales_a) - np.log2(scales_b)
                if np.isnan(lfc).any():
                    warnings.warn("NaN values appeared in LFCs")

                pgs = np.nanmean(np.abs(lfc) >= delta, axis=0)
                de_probas[training, size_ix, exp, :] = pgs
    np.save(file=filename, arr=de_probas)
    return de_probas


def multi_train_estimates(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    delta: float = 0.5,
    n_trainings: int = 5,
    n_picks: int = 25,
    n_samples: int = 500,
    label_a=0,
    label_b=1,
    importance_sampling=False,
    normalized_means=None
):
    """

    """
    n_examples, n_genes = dataset.X.shape
    if os.path.exists(filename):
        return pd.read_pickle(filename)

    n_sizes = len(sizes)
    # de_probas = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes))
    # lfc_means = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes))
    # lfc_stds = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes))
    # hdis64 = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes, 2))
    # hdis99 = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes, 2))

    #     lfcs = np.zeros((n_trainings, N_SIZES, n_picks, dataset.nb_genes, 3*n_samples))
    dfs_li = []
    local_lfc_gt = None
    local_is_de = None
    for training in range(n_trainings):
        my_vae, my_trainer = train_model(
            mdl_class, dataset, mdl_params, train_params, train_fn_params
        )
        post = my_trainer.test_set
        test_indices = post.data_loader.sampler.indices
        # train_samples = np.random.permutation(train_indices)
        post = my_trainer.create_posterior(
            model=my_vae, gene_dataset=dataset, indices=test_indices
        )
        outputs = post.get_latents(
            n_samples=n_samples, other=True, device="cpu"
        )
        z, labels, scales = outputs["z"], outputs["label"], outputs["scale"]
        weights = softmax(outputs["log_probas"], axis=0)

        for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
            for exp in range(n_picks):
                labels = labels.squeeze()
                # Sampling cells
                where_a = np.where(labels == label_a)[0]
                where_b = np.where(labels == label_b)[0]
                where_a = where_a[np.random.choice(len(where_a), size=size)]
                where_b = where_b[np.random.choice(len(where_b), size=size)]
                # Sampling z observations
                scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
                scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
                if importance_sampling:
                    weights_a = weights[:, where_a].reshape((-1)) / len(where_a)
                    weights_b = weights[:, where_b].reshape((-1)) / len(where_b)
                else:
                    weights_a = None
                    weights_b = None
                if normalized_means is not None:
                    overall_idx_a = test_indices[where_a]
                    overall_idx_b = test_indices[where_b]
                    h_a = normalized_means[overall_idx_a].reshape((-1, 1, n_genes))
                    h_b = normalized_means[overall_idx_b].reshape((1, -1, n_genes))
                    lfc_dist = (np.log2(h_a) - np.log2(h_b)).reshape((-1, n_genes))
                    local_lfc_gt = lfc_dist.mean(0)
                    local_is_de = (np.abs(lfc_dist) >= delta).mean(0)
                scales_a, scales_b = demultiply(
                    arr1=scales_a,
                    arr2=scales_b,
                    factor=3,
                    weights_a=weights_a,
                    weights_b=weights_b
                )
                lfc = np.log2(scales_a) - np.log2(scales_b)
                assert lfc.shape[1] == dataset.nb_genes
                if np.isnan(lfc).any():
                    warnings.warn("NaN values appeared in LFCs")

                pgs = np.nanmean(np.abs(lfc) >= delta, axis=0)
                lfc_mean = np.nanmean(lfc, axis=0)
                lfc_std = np.nanstd(lfc, axis=0)
                hdi64 = compute_hdi(lfc, credible_interval=0.64)
                hdi25 = compute_hdi(lfc, credible_interval=0.25)
                hdi50 = compute_hdi(lfc, credible_interval=0.50)
                hdi75 = compute_hdi(lfc, credible_interval=0.75)
                hdi95 = compute_hdi(lfc, credible_interval=0.95)
                hdi99 = compute_hdi(lfc, credible_interval=0.99)

                df = pd.DataFrame(
                    dict(
                        de_proba=pgs,
                        lfc_mean=lfc_mean,
                        lfc_std=lfc_std,
                        hdi25_low=hdi25[:, 0],
                        hdi25_high=hdi25[:, 1],
                        hdi50_low=hdi50[:, 0],
                        hdi50_high=hdi50[:, 1],
                        hdi75_low=hdi75[:, 0],
                        hdi75_high=hdi75[:, 1],
                        hdi95_low=hdi95[:, 0],
                        hdi95_high=hdi95[:, 1],
                        hdi64_low=hdi64[:, 0],
                        hdi64_high=hdi64[:, 1],
                        hdi99_low=hdi99[:, 0],
                        hdi99_high=hdi99[:, 1],
                    )
                ).assign(
                    experiment=lambda x: exp,
                    sample_size=lambda x: size,
                    training=lambda x: training,
                    gene=np.arange(dataset.nb_genes),
                    lfc_gt=local_lfc_gt,
                    is_de=local_is_de,
                )
                dfs_li.append(df)
    df_res = pd.concat(dfs_li, ignore_index=True)

    # de_probas[training, size_ix, exp, :] = pgs
    # lfc_means[training, size_ix, exp, :] = lfc_mean
    # lfc_stds[training, size_ix, exp, :] = lfc_std
    # hdis64[training, size_ix, exp, :] = hdi64
    # hdis99[training, size_ix, exp, :] = hdi99
    df_res.to_pickle(filename)
    return df_res
