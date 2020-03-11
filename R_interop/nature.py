import rpy2.robjects as ro
import pandas as pd
import rpy2.robjects.numpy2ri
import warnings
import numpy as np
from rpy2.rinterface import RRuntimeWarning


class DEClass:
    def __init__(
        self,
        A: int,
        B: int,
        data: str,
        labels: str,
        normalized_means: str,
        delta: float,
        cluster: tuple,
        path_to_scripts: str,
        batches: str = None,
    ):
        """
        A: number of cells in the first cluster
        B: number of cells in the second cluster
        data: dataset to look at
        labels: clusters
        cluster: list that tells which cluster to test ex. (0, 4)
        """
        self.A = A
        self.B = B
        self.data = data
        self.labels = labels
        self.batches = batches
        self.cluster = cluster
        # loading libraries
        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        rpy2.robjects.numpy2ri.activate()
        ro.r["library"]("RcppCNPy")
        ro.r["library"]("BiocParallel")
        ro.r("BiocParallel::register(BiocParallel::MulticoreParam())")

        ro.r.assign("path_to_scripts", path_to_scripts)

        self.X_train = np.load(self.data)
        n_samples, n_genes = self.X_train.shape
        self.c_train = np.loadtxt(self.labels)
        # loading data
        ro.r(
            str("""fmat <- npyLoad("*""")[:-1]
            + self.data
            + str("""*", "integer")""")[1:]
        )
        ro.r(str("""cmat <- read.table("*""")[:-1] + self.labels + str("""*")""")[1:])

        if self.batches is not None:
            ro.r(
                str("""batch_indices <- read.table("*""")[:-1]
                + self.batches
                + str("""*")""")[1:]
            )
        # ro.r("cmat$V2 <- factor(cmat$V1)")

        # computing data mask
        set_a = np.where(self.c_train == self.cluster[0])[0]
        subset_a = np.random.choice(set_a, self.A, replace=False) + 1
        set_b = np.where(
            (self.c_train == self.cluster[1])
            * (
                ~np.isin(np.arange(len(self.c_train)), subset_a)
            )  # avoid using same samples for negative controls
        )[0]
        subset_b = np.random.choice(set_b, self.B, replace=False) + 1

        self.lfc_gt = None
        self.is_de = None
        if normalized_means is not None:
            self.normalized_means = np.load(normalized_means)
            h_a = self.normalized_means[subset_a].reshape((-1, 1, n_genes))
            h_b = self.normalized_means[subset_b].reshape((1, -1, n_genes))
            lfc_dist = (np.log2(h_a) - np.log2(h_b)).reshape((-1, n_genes))
            self.lfc_gt = lfc_dist.mean(0)
            self.is_de = (np.abs(lfc_dist) >= delta).mean(0)

        stochastic_set = np.hstack((subset_a, subset_b))

        # Option default
        # f = np.array([a in stochastic_set for a in np.arange(self.X_train.shape[0])])
        # nr, nc = f[:, np.newaxis].shape
        # f_r = ro.r.matrix(f[:, np.newaxis], nrow=nr, ncol=nc)
        # ro.r.assign("f_", f_r)
        # ro.r("f <- as.integer(rownames(cmat[f_,]))")
        # ro.r("local_fmat <- fmat[f, ]")
        # ro.r("local_cmat <- cmat[f, ]")
        # ro.r("local_cmat$V3 <- factor(local_cmat$V1)")

        # # Other option
        # Mask denoting if cell is kept
        ro.r.assign("f", stochastic_set)
        ro.r("local_fmat <- t(fmat[f,])")
        ro.r("local_fmat <- as.data.frame(local_fmat)")
        if self.A == self.B:
            print("Negative control task; artificial change of labels ...")
            ro.r.assign("a_indices", subset_a)
            ro.r.assign("b_indices", subset_b)
            ro.r("cmat[a_indices,] <- 1")
            ro.r("cmat[b_indices,] <- 2")
        ro.r("local_cmat <- factor(cmat[f,])")
        ro.r("local_batch <- factor(batch_indices[f,])")

        if self.batches is None:
            ro.r("L <- list(count=local_fmat, condt=local_cmat)")
        else:
            ro.r("L <- list(count=local_fmat, condt=local_cmat, batch=local_batch)")

    def fit(self):
        pass


class NEdgeRLTRT(DEClass):
    def __init__(
        self,
        A: int,
        B: int,
        data: str,
        labels: str,
        normalized_means: str,
        delta: float,
        cluster: tuple,
        path_to_scripts: str,
        batches: str = None,
    ):
        super().__init__(
            A=A,
            B=B,
            data=data,
            labels=labels,
            normalized_means=normalized_means,
            delta=delta,
            cluster=cluster,
            path_to_scripts=path_to_scripts,
            batches=batches,
        )

    def fit(self):
        ro.r("script_path <- paste(path_to_scripts, 'apply_edgeRLRT.R', sep='/')")
        ro.r("source(script_path)")
        if self.batches is None:
            ro.r("res <- run_edgeRLRT(L)")
        else:
            ro.r("res <- run_edgeRLRT_multibatch(L)")
        res = pd.DataFrame(ro.r("res$df")).assign(lfc_gt=self.lfc_gt, is_de=self.is_de)
        return res


class NEdgeRLTRTRobust(DEClass):
    def __init__(
        self,
        A: int,
        B: int,
        data: str,
        labels: str,
        normalized_means: str,
        delta: float,
        cluster: tuple,
        path_to_scripts: str,
    ):
        super().__init__(
            A=A,
            B=B,
            data=data,
            labels=labels,
            normalized_means=normalized_means,
            delta=delta,
            cluster=cluster,
            path_to_scripts=path_to_scripts,
        )

    def fit(self):
        ro.r("script_path <- paste(path_to_scripts, 'apply_edgeRLRTrobust.R', sep='/')")
        ro.r("source(script_path)")
        ro.r("res <- run_edgeRLRTrobust(L)")
        res = pd.DataFrame(ro.r("res$df")).assign(lfc_gt=self.lfc_gt, is_de=self.is_de)
        return res


class NDESeq2(DEClass):
    def __init__(
        self,
        A: int,
        B: int,
        data: str,
        labels: str,
        normalized_means: str,
        delta: float,
        cluster: tuple,
        path_to_scripts: str,
        lfc_threshold: float = 0.5,
        batches: str = None,
    ):
        ro.r.assign("lfc_threshold", lfc_threshold)

        super().__init__(
            A=A,
            B=B,
            data=data,
            labels=labels,
            normalized_means=normalized_means,
            delta=delta,
            cluster=cluster,
            path_to_scripts=path_to_scripts,
            batches=batches,
        )

    def fit(self):
        ro.r("script_path <- paste(path_to_scripts, 'apply_DESeq2.R', sep='/')")
        ro.r("source(script_path)")
        if self.batches is None:
            ro.r("res <- run_DESeq2(L, lfcThreshold=lfc_threshold)")
        else:
            ro.r("res <- run_DESeq2_multibatch(L, lfcThreshold=lfc_threshold)")
        res = pd.DataFrame(ro.r("res$df")).assign(lfc_gt=self.lfc_gt, is_de=self.is_de)
        return res


class NMASTcpm(DEClass):
    def __init__(
        self,
        A: int,
        B: int,
        data: str,
        labels: str,
        normalized_means: str,
        delta: float,
        cluster: tuple,
        path_to_scripts: str,
        batches: str = None,
    ):
        super().__init__(
            A=A,
            B=B,
            data=data,
            labels=labels,
            normalized_means=normalized_means,
            delta=delta,
            cluster=cluster,
            path_to_scripts=path_to_scripts,
            batches=batches,
        )

    def fit(self):
        ro.r("script_path <- paste(path_to_scripts, 'apply_MASTcpm.R', sep='/')")
        ro.r("source(script_path)")

        if self.batches is None:
            ro.r("res <- run_MASTcpm(L)")
        else:
            ro.r("res <- run_MASTcpm_multibatch(L)")
        # return ro.r("res")
        res = pd.DataFrame(ro.r("res$df"))
        return res


# class NSCDE(DEClass):
#     def __init__(
#             self,
#             A: int,
#             B: int,
#             data: str,
#             labels: str,
#             cluster: tuple,
#             path_to_scripts: str,
#     ):
#             super().__init__(
#                 A=A,
#                 B=B,
#                 data=data,
#                 labels=labels,
#                 cluster=cluster,
#                 path_to_scripts=path_to_scripts,
#             )
#
#     def fit(self):
#         ro.r("script_path <- paste(path_to_scripts, 'apply_SCDE.R', sep='/')")
#         ro.r("source(script_path)")
#         ro.r("res <- run_SCDE(L)")
#         # return ro.r("res")
#         return pd.DataFrame(ro.r("res$df")), ro.r("res")

