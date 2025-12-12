import torch
import numpy as np
from torch.optim import LBFGS
import gc
from natsort import natsorted
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import pyranges as pr
from scipy.stats import combine_pvalues

def initialize_weights(L,
                       initialize_method='He'):
    """
    Initialize weights for the linkage matrix using different methods.
    
    Args:
        L (torch.Tensor): The linkage matrix to initialize
        initialize_method (str, optional): Method to initialize weights. Options: 'He', 'Xavier', or None. Defaults to 'He'.
    """
    if initialize_method=='He':
        nn.init.kaiming_normal_(L, nonlinearity='linear')

    if initialize_method=='Xavier':
        nn.init.xavier_normal_(L)

    if initialize_method is None:
        return(L)

def closure(G,
            P,
            L,
            optimizer,
            lambda_l2,
            losses):
    """
    Create a closure function for LBFGS optimizer to compute loss and gradients.
    
    Args:
        G (torch.Tensor): Gene expression matrix
        P (torch.Tensor): Peak accessibility matrix
        L (torch.Tensor): Linkage matrix
        optimizer (torch.optim): Optimizer instance
        lambda_l2 (float): L2 regularization parameter
        losses (list): List to store loss values
    
    Returns:
        function: Closure function for optimizer
    """
    def closure_fn():
        optimizer.zero_grad()

        predictions1 = torch.matmul(L, P)
        predictions2 = torch.matmul(L.T, G)

        criterion = torch.nn.MSELoss()
        loss = criterion(predictions1, G) + \
               criterion(predictions2, P) + \
               lambda_l2 * torch.norm(L, 2)
        
        losses.append(loss.item())
        loss.backward()

        torch.cuda.empty_cache()
        gc.collect()
        return loss

    return closure_fn


def plot_losses(losses,
                chrs):
    """
    Plot training losses for each chromosome.
    
    Args:
        losses (list): List of loss values for each chromosome
        chrs (list): List of chromosome names
    """
    num_plots = len(losses)
    ncols = 5
    nrows = num_plots // ncols + (num_plots % ncols > 0)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2*nrows))

    for idx, loss_values in enumerate(losses):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]
        ax.plot(loss_values, label=None)
        ax.set_title(chrs[idx])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    for idx in range(num_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.show()


def linkage(rna_adata,
             atac_adata,
             meta,
             min_cell=5,
             lr=0.1,
             max_iter=100,
             lambda_l2=0.1,
             plot=True,
             initialize_method='He',
             normalize='col',
             exclude_chr='chrM',
             exclude_gene=None,
             exclude_peak=None,
             downsample=None,
             tolerance_grad=1e-7,
             tolerance_change=1e-9,
             history_size=100,
             verbose=False):
    """
    Calculate linkage between RNA and ATAC data.
    
    Args:
        rna_adata (AnnData): RNA expression data
        atac_adata (AnnData): ATAC accessibility data
        meta (pd.DataFrame): Metadata containing gene and peak information
        min_cell (int, optional): Minimum number of cells expressing a feature. Defaults to 5.
        lr (float, optional): Learning rate for optimization. Defaults to 0.1.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        lambda_l2 (float, optional): L2 regularization parameter. Defaults to 0.1.
        plot (bool, optional): Whether to plot losses. Defaults to True.
        initialize_method (str, optional): Weight initialization method. Defaults to 'He'.
        normalize (str, optional): Normalization method ('row' or 'col'). Defaults to 'col'.
        exclude_chr (str, optional): Chromosome to exclude. Defaults to 'chrM'.
        exclude_gene (list, optional): Genes to exclude. Defaults to None.
        exclude_peak (list, optional): Peaks to exclude. Defaults to None.
        downsample (int, optional): Number of cells to downsample. Defaults to None.
    
    Returns:
        list: List of linkage matrices for each chromosome
    """
    rna_cells = rna_adata.obs.index.tolist()
    atac_cells = atac_adata.obs.index.tolist()

    if sorted(rna_cells) != sorted(atac_cells):
        raise ValueError('The RNA and ATAC data is not paired!')

    if rna_cells != atac_cells:
        rna_cells_sorted = sorted(rna_cells)
        atac_cells_sorted = sorted(atac_cells)
        rna_adata = rna_adata[rna_cells_sorted, :]
        atac_adata = atac_adata[atac_cells_sorted, :]

    if downsample is not None:

        if downsample > rna_adata.n_obs:
            downsample = rna_adata.n_obs

        print('Using ' + str(downsample) + ' cells to calculate linkage.')
        random_indices = np.random.choice(rna_adata.n_obs, size=downsample, replace=False)
        rna_adata = rna_adata[random_indices].copy()
        atac_adata = atac_adata[random_indices].copy()

    atac_adata.var.index = atac_adata.var.index.str.replace(':', '-')

    chrs = natsorted([element for element in np.unique(meta[3].tolist()) if element.startswith('chr')])

    if exclude_chr is not None:
        chrs = [chr for chr in chrs if chr not in exclude_chr]

    gene_to_index = {gene: idx for idx, gene in enumerate(rna_adata.var.index)}
    peak_to_index = {peak: idx for idx, peak in enumerate(atac_adata.var.index)}

    linkage = []
    losses_list = []

    for chr in tqdm(chrs):

        chr_gene = np.unique(
            meta.iloc[np.where((meta[3].to_numpy() == chr) & (meta[2].to_numpy() == 'Gene Expression'))][1].to_numpy())
        #chr_peak = np.unique( meta.iloc[np.where((meta[3].to_numpy() == chr) & (meta[2].to_numpy() == 'Peaks'))][1].to_numpy())
        #chr_peak = [element.replace(':', '-') for element in chr_peak]
        chr_peak = np.array(natsorted([peak for peak in atac_adata.var.index if peak.startswith(str(chr + '-'))]))
        chr_gene = np.sort(np.array(list(set(chr_gene).intersection(set(rna_adata.var.index)))))
        #chr_peak = np.sort(np.array(list(set(chr_peak).intersection(set(atac_adata.var.index)))))

        gene_indices = [gene_to_index[gene] for gene in chr_gene if gene in gene_to_index]
        chr_gene_exp_data = rna_adata.X[:, gene_indices].T.toarray()

        peak_indices = [peak_to_index[peak] for peak in chr_peak if peak in peak_to_index]
        chr_peak_exp_data = atac_adata.X[:, peak_indices].T.toarray()

        select_gene = chr_gene[np.where(np.sum(chr_gene_exp_data > 0, axis=1) > min_cell)[0]]
        select_peak = chr_peak[np.where(np.sum(chr_peak_exp_data > 0, axis=1) > min_cell)[0]]

        if exclude_gene is not None:
            select_gene = [gene for gene in select_gene if gene not in exclude_gene]

        if exclude_peak is not None:
            select_peak = [peak for peak in select_peak if peak not in exclude_peak]

        select_gene_to_index = {gene: idx for idx, gene in enumerate(chr_gene)}
        select_gene_indices = [select_gene_to_index[gene] for gene in select_gene if gene in gene_to_index]
        chr_gene_exp_data = chr_gene_exp_data[select_gene_indices, :]

        select_peak_to_index = {peak: idx for idx, peak in enumerate(chr_peak)}
        select_peak_indices = [select_peak_to_index[peak] for peak in select_peak if peak in peak_to_index]
        chr_peak_exp_data = chr_peak_exp_data[select_peak_indices, :]

        filter_cell = np.unique(np.concatenate(
            (np.where(np.sum(chr_gene_exp_data, axis=0) == 0)[0], np.where(np.sum(chr_peak_exp_data, axis=0) == 0)[0])))

        if len(filter_cell) > 0:
            chr_gene_exp_data = np.delete(chr_gene_exp_data, filter_cell, axis=1)
            chr_peak_exp_data = np.delete(chr_peak_exp_data, filter_cell, axis=1)

        # Check if we have valid genes and peaks after filtering
        # This prevents the RuntimeError that occurs when the linkage matrix L becomes empty
        if verbose:
            print(f"Processing chromosome {chr}: {chr_gene_exp_data.shape[0]} genes, {chr_peak_exp_data.shape[0]} peaks, {chr_gene_exp_data.shape[1]} cells")
        
        if chr_gene_exp_data.shape[0] == 0:
            print(f"Warning: No genes found for chromosome {chr} after filtering. Skipping.")
            continue
            
        if chr_peak_exp_data.shape[0] == 0:
            print(f"Warning: No peaks found for chromosome {chr} after filtering. Skipping.")
            continue
            
        if chr_gene_exp_data.shape[1] == 0:
            print(f"Warning: No cells found for chromosome {chr} after filtering. Skipping.")
            continue

        # Skip chromosomes with too few genes or peaks (may cause numerical instability)
        if chr_gene_exp_data.shape[0] < 2:
            print(f"Warning: Too few genes ({chr_gene_exp_data.shape[0]}) for chromosome {chr}. Skipping.")
            continue
            
        if chr_peak_exp_data.shape[0] < 2:
            print(f"Warning: Too few peaks ({chr_peak_exp_data.shape[0]}) for chromosome {chr}. Skipping.")
            continue

        # Clear GPU cache before processing each chromosome
        torch.cuda.empty_cache()
        gc.collect()
        
        G = torch.from_numpy(chr_gene_exp_data).cuda(0)
        P = torch.from_numpy(chr_peak_exp_data).cuda(0)
        
        if normalize == 'row':
            G = (G - torch.mean(G, dim=1, keepdim=True)) / torch.std(G, dim=1, keepdim=True)
            P = (P - torch.mean(P, dim=1, keepdim=True)) / torch.std(P, dim=1, keepdim=True)

        if normalize == 'col':
            G = (G - torch.mean(G, dim=0)) / torch.std(G, dim=0)
            P = (P - torch.mean(P, dim=0)) / torch.std(P, dim=0)

        L = torch.rand((G.shape[0], P.shape[0]), requires_grad=True, device="cuda")
        initialize_weights(L, initialize_method=initialize_method)

        losses = []

        optimizer = LBFGS([L], lr=lr, max_iter=max_iter, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        closure_fn = closure(G, P, L, optimizer, lambda_l2, losses)
        try:
            optimizer.step(closure_fn)
        except RuntimeError as e:
            print(f"Warning: Optimization failed for chromosome {chr}: {str(e)}")
            print(f"Chromosome {chr} dimensions - Genes: {G.shape[0]}, Peaks: {P.shape[0]}, Cells: {G.shape[1]}")
            continue
        losses_list.append(losses)
        result = pd.DataFrame(L.cpu().detach().numpy(), index=select_gene, columns=select_peak)
        linkage.append(result)
        torch.cuda.empty_cache()
        gc.collect()

    if plot:
        plot_losses(losses_list, chrs)

    return linkage

def calculate_median_mad(values):
    """
    Calculate median and MAD (Median Absolute Deviation) for each row.
    
    Args:
        values (np.ndarray): Input matrix
    
    Returns:
        tuple: (median values, MAD values)
    """
    median_vals = np.median(values, axis=1, keepdims=True)
    abs_deviations = np.abs(values - median_vals)
    mad_vals = np.median(abs_deviations, axis=1, keepdims=True) * 1.4826
    return median_vals, mad_vals


def calculate_pvalue(l,
                      meta_data,
                      upstream=500000,
                      downstream=500000,
                      method='pearson'):
    """
    Calculate p-values for gene-peak linkages.
    
    Args:
        l (list): List of linkage matrices
        meta_data (pd.DataFrame): Metadata containing gene and peak information
        upstream (int, optional): Upstream distance for peak-gene pairs. Defaults to 500000.
        downstream (int, optional): Downstream distance for peak-gene pairs. Defaults to 500000.
        method (str, optional): Method to combine p-values. Defaults to 'pearson'.
    
    Returns:
        pd.DataFrame: DataFrame containing gene-peak pairs with p-values
    """
    melted_chrs = []
    for l_chr in tqdm(l):
        linkage_values = l_chr.values
        n_genes, n_peaks = linkage_values.shape
        gene_median, gene_mad = calculate_median_mad(linkage_values)
        #gene_p_vals = norm.sf(linkage_values, loc=gene_median, scale=gene_mad)
        abs_values = np.abs(linkage_values - gene_median)
        gene_p_vals = 2 * norm.sf(abs_values, loc=0, scale=gene_mad)
        gene_p_vals_adj = np.zeros_like(gene_p_vals)
        for i in range(n_genes):
            gene_p_vals_adj[i] = multipletests(gene_p_vals[i], method='bonferroni')[1]
        linkage_values_T = linkage_values.T
        peak_median, peak_mad = calculate_median_mad(linkage_values_T)
        #peak_p_vals = norm.sf(linkage_values_T, loc=peak_median, scale=peak_mad)
        abs_values = np.abs(linkage_values_T - peak_median)
        peak_p_vals = norm.sf(abs_values, loc=peak_median, scale=peak_mad)
        peak_p_vals_adj = np.zeros_like(peak_p_vals)
        for i in range(n_peaks):
            peak_p_vals_adj[i] = multipletests(peak_p_vals[i], method='bonferroni')[1]
        peak_p_vals_adj = peak_p_vals_adj.T
        gene_indices = np.arange(n_genes).repeat(n_peaks)
        peak_indices = np.tile(np.arange(n_peaks), n_genes)
        melted_chr = l_chr.stack().reset_index()
        melted_chr.columns = ['Gene', 'Peak', 'Value']
        melted_chr['P.adj_Gene'] = gene_p_vals_adj[gene_indices, peak_indices]
        melted_chr['P.adj_Peak'] = peak_p_vals_adj[gene_indices, peak_indices]
        gene_meta = meta_data[meta_data[1].isin(l_chr.index)].copy()
        gene_meta.columns = ['Ensemble', 'Gene', 'Type', 'Chromosome', 'Start', 'End']
        peak_info = pd.DataFrame({
            'Peak': l_chr.columns,
            'Chr_Start_End': l_chr.columns
        })
        peak_split = peak_info['Chr_Start_End'].str.split('-', expand=True)
        peak_info[['Chromosome', 'Start', 'End']] = peak_split
        peak_info['Start'] = pd.to_numeric(peak_info['Start'], errors='coerce')
        peak_info['End'] = pd.to_numeric(peak_info['End'], errors='coerce')
        range_peak = pr.PyRanges(peak_info[['Chromosome', 'Start', 'End', 'Peak']])
        valid_pairs = set()
        for _, row in gene_meta.iterrows():
            gene = row['Gene']
            gene_range = pr.PyRanges(pd.DataFrame({
                'Chromosome': [row['Chromosome']],
                'Start': [row['Start'] - downstream],
                'End': [row['End'] + upstream],
                'Gene': [gene]
            }))
            range_intersect_peak = range_peak.overlap(gene_range)
            if len(range_intersect_peak) > 0:
                intersect_peak = range_intersect_peak.df['Peak'].tolist()
                valid_pairs.update((gene, peak) for peak in intersect_peak)
        gene_peak_pairs = pd.DataFrame(list(valid_pairs), columns=['Gene', 'Peak'])
        melted_chr_distance = melted_chr.merge(gene_peak_pairs, on=['Gene', 'Peak'])
        melted_chr_distance = melted_chr_distance.drop_duplicates(subset=['Gene', 'Peak'])
        p_values = np.column_stack([melted_chr_distance['P.adj_Gene'], melted_chr_distance['P.adj_Peak']])
        combine_p_vals = np.array([combine_pvalues(p, method).pvalue for p in p_values])
        melted_chr_distance['Combine_p_value'] = combine_p_vals
        melted_chrs.append(melted_chr_distance)
    result = pd.concat(melted_chrs, ignore_index=True)
    return (result)












