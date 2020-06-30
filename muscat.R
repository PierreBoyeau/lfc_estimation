library(cowplot)
library(dplyr)
library(reshape2)
library(muscat)
library(purrr)
library(scater)
library(SingleCellExperiment)

data(sce)
ref <- prepSim(sce, verbose = FALSE)
# only samples from `ctrl` group are retained
table(ref$sample_id)


# cell parameters: library sizes
sub <- assay(sce[rownames(ref), colnames(ref)])
all.equal(exp(ref$offset), as.numeric(colSums(sub)))

head(rowData(ref))


nc <- 10000;
nk <- 3;  # Number of subpopulations
ns <- 3;  # Number of replicates
p_dd <- diag(6)[1, ];   # p_dd = c(1,0,...0), i.e., 10% of EE genes

p_dd <- c(0.3, 0.3, 0.25, 0.05, 0.05, 0.05);
sum(p_dd)

ng <- 1000;  # Number of genes

# simulated 10% EE genes
sim <- simData(
    ref, 
    p_dd = p_dd,
    nk = nk, 
    ns = ns, 
    nc = nc,
    ng = ng, 
    force = TRUE
)


# number of cells per sample and subpopulation
table(sim$sample_id, sim$cluster_id)

metadata(sim)$ref_sids
colData(sim)

gene_info <- metadata(sim)$gene_info;
counts <- assay(sim)
cell_info <- colData(sim)

typeof(gene_info)
str(gene_info)
gene_info$cluster_id
write.csv(gene_info, "/home/pierre/lfc_estimation/muscat_gene_info.csv")
logFCs <- gene_info$logFC;

write.csv(counts, file = "/home/pierre/lfc_estimation/muscat_counts.csv")

write.csv(cell_info, file = "/home/pierre/lfc_estimation/muscat_cell_info.csv")