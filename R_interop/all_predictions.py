import time
import numpy as np
from tqdm import tqdm
import os
import pickle
from . import NDESeq2, NEdgeRLTRT, MAST, NMASTcpm, NEdgeRLTRTRobust


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


def all_predictions(
    filename,
    n_genes,
    n_picks,
    sizes,
    data_path,
    labels_path,
    size_b=None,
    label_a=0,
    label_b=1,
    normalized_means=None,
    delta=None,
    path_to_scripts=None,
    lfc_threshold: float = 0.5,
    all_nature=True,
    mast_cmat_key="V31",
    batches: str = None,
):
    if os.path.exists(filename):
        return load_pickle(filename)
    n_sizes = len(sizes)

    results = dict()
    # DESeq2
    lfcs_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    times_deseq2 = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            timer = time.time()
            deseq_inference = NDESeq2(
                A=size,
                B=b_size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
                normalized_means=normalized_means,
                delta=delta,
                path_to_scripts=path_to_scripts,
                lfc_threshold=lfc_threshold,
                batches=batches,
            )
            try:
                res_df = deseq_inference.fit()
                timer = time.time() - timer
                times_deseq2[size_ix, exp] = timer
                lfcs_deseq2[size_ix, exp, :] = res_df["lfc"].values
                pvals_deseq2[size_ix, exp, :] = res_df["padj"].values
                pvals_true_deseq2[size_ix, exp, :] = res_df["pval"].values
            except Exception as e:
                print(e)
    deseq_res = dict(
        lfc=lfcs_deseq2.squeeze(),
        pval=pvals_deseq2.squeeze(),
        pval_true=pvals_true_deseq2.squeeze(),
        time=times_deseq2,
    )
    results["deseq2"] = deseq_res
    save_pickle(data=results, filename=filename)

    # EdgeR
    lfcs_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    times_edge_r = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            timer = time.time()
            deseq_inference = NEdgeRLTRT(
                A=size,
                B=b_size,
                data=data_path,
                labels=labels_path,
                normalized_means=normalized_means,
                delta=delta,
                cluster=(label_a, label_b),
                path_to_scripts=path_to_scripts,
                batches=batches,
            )
            try:
                res_df = deseq_inference.fit()
                timer = time.time() - timer
                times_edge_r[size_ix, exp] = timer
                lfcs_edge_r[size_ix, exp, :] = res_df["lfc"].values
                pvals_edge_r[size_ix, exp, :] = res_df["padj"].values
                pvals_true_edge_r[size_ix, exp, :] = res_df["pval"].values
            except Exception as e:
                print(e)

    edger_res = dict(
        lfc=lfcs_edge_r.squeeze(),
        pval=pvals_edge_r.squeeze(),
        pval_true=pvals_true_edge_r.squeeze(),
        time=times_edge_r,
    )
    results["edger"] = edger_res
    save_pickle(data=results, filename=filename)

    # MAST
    lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    var_lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    pvals_mast = np.zeros((n_sizes, n_picks, n_genes))
    times_mast = np.zeros((n_sizes, n_picks))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        b_size = size_b if size_b is not None else size
        print("Size A {} and B {}".format(size, b_size))
        for exp in range(n_picks):
            if all_nature:
                timer = time.time()
                mast_inference = NMASTcpm(
                    A=size,
                    B=b_size,
                    data=data_path,
                    labels=labels_path,
                    normalized_means=normalized_means,
                    delta=delta,
                    cluster=(label_a, label_b),
                    path_to_scripts=path_to_scripts,
                    batches=batches,
                )
                try:
                    res_df = mast_inference.fit()
                    timer = time.time() - timer
                    times_mast[size_ix, exp] = timer
                    print(res_df.info())
                    # var_lfcs_mast[size_ix, exp, :] = res_df["varLogFC"].values
                    lfcs_mast[size_ix, exp, :] = res_df["lfc"].values
                    pvals_mast[size_ix, exp, :] = res_df["pval"].values
                except Exception as e:
                    print(e)

            else:
                timer = time.time()
                mast_inference = MAST(
                    A=size,
                    B=b_size,
                    data=data_path,
                    labels=labels_path,
                    cluster=(label_a, label_b),
                    local_cmat_key=mast_cmat_key,
                )
                try:
                    res_df = mast_inference.fit(return_fc=True)
                    timer = time.time() - timer
                    times_mast[size_ix, exp] = timer
                    lfcs_mast[size_ix, exp, :] = res_df["lfc"].values
                    pvals_mast[size_ix, exp, :] = res_df["pval"].values
                except Exception as e:
                    print(e)
    mast_res = dict(
        lfc=lfcs_mast.squeeze(),
        pval=pvals_mast.squeeze(),
        var_lfc=var_lfcs_mast,
        time=times_mast,
    )
    results["mast"] = mast_res
    save_pickle(data=results, filename=filename)
    return results


def all_de_predictions(dict_results, significance_level, delta):
    """

    :param dict_results: dictionnary of dictionnary with hierarchical keys:
        algorithm:
            lfc
            pval
    :param significance_level:
    :param delta:
    :return:
    """
    for algorithm_name in dict_results:
        my_pvals = dict_results[algorithm_name]["pval"]
        my_pvals[np.isnan(my_pvals)] = 1.0

        my_lfcs = dict_results[algorithm_name]["lfc"]
        my_lfcs[np.isnan(my_lfcs)] = 0.0

        if algorithm_name == "deseq2":
            is_de = my_pvals <= significance_level

        elif algorithm_name == "edger" or algorithm_name == "edger_robust":
            is_de = my_pvals <= significance_level

        elif algorithm_name == "mast":
            is_de = (my_pvals <= significance_level) * (np.abs(my_lfcs) >= delta)
        else:
            raise KeyError("No DE policy for {}".format(algorithm_name))
        dict_results[algorithm_name]["is_de"] = is_de
    return dict_results
