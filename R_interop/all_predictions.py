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
    label_a=0,
    label_b=1,
    normalized_means=None,
    delta=None,
    path_to_scripts=None,
    lfc_threshold: float = 0.5,
    all_nature=True,
):
    if os.path.exists(filename):
        return load_pickle(filename)
    n_sizes = len(sizes)

    results = dict()
    # DESeq2
    lfcs_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            deseq_inference = NDESeq2(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
                normalized_means=normalized_means,
                delta=delta,
                path_to_scripts=path_to_scripts,
                lfc_threshold=lfc_threshold,
            )
            res_df = deseq_inference.fit()
            lfcs_deseq2[size_ix, exp, :] = res_df["lfc"].values
            pvals_deseq2[size_ix, exp, :] = res_df["padj"].values
            pvals_true_deseq2[size_ix, exp, :] = res_df["pval"].values
    deseq_res = dict(lfc=lfcs_deseq2.squeeze(), pval=pvals_deseq2.squeeze(), pval_true=pvals_true_deseq2.squeeze())
    results["deseq2"] = deseq_res
    save_pickle(data=results, filename=filename)

    # EdgeR
    lfcs_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_true_edge_r = np.zeros((n_sizes, n_picks, n_genes))

    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            deseq_inference = NEdgeRLTRT(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                normalized_means=normalized_means,
                delta=delta,
                cluster=(label_a, label_b),
                path_to_scripts=path_to_scripts,
            )
            res_df = deseq_inference.fit()
            lfcs_edge_r[size_ix, exp, :] = res_df["lfc"].values
            pvals_edge_r[size_ix, exp, :] = res_df["padj"].values
            pvals_true_edge_r[size_ix, exp, :] = res_df["pval"].values

    edger_res = dict(lfc=lfcs_edge_r.squeeze(), pval=pvals_edge_r.squeeze(), pval_true=pvals_true_edge_r.squeeze())
    results["edger"] = edger_res
    save_pickle(data=results, filename=filename)


    # MAST
    lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    var_lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    pvals_mast = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            if all_nature:
                mast_inference = NMASTcpm(
                    A=size,
                    B=size,
                    data=data_path,
                    labels=labels_path,
                    normalized_means=normalized_means,
                    delta=delta,
                    cluster=(label_a, label_b),
                    path_to_scripts=path_to_scripts,
                )
                res_df = mast_inference.fit()
                print(res_df.info())
                var_lfcs_mast[size_ix, exp, :] = res_df["varLogFC"].values
                lfcs_mast[size_ix, exp, :] = res_df["logFC"].values

            else:
                mast_inference = MAST(
                    A=size,
                    B=size,
                    data=data_path,
                    labels=labels_path,
                    cluster=(label_a, label_b),
                )
                res_df = mast_inference.fit(return_fc=True)
                lfcs_mast[size_ix, exp, :] = res_df["lfc"].values
            pvals_mast[size_ix, exp, :] = res_df["pval"].values
    mast_res = dict(
        lfc=lfcs_mast.squeeze(), pval=pvals_mast.squeeze(), var_lfc=var_lfcs_mast
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
