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
from joblib import Parallel, delayed
import pyranges as pr
from scipy.stats import combine_pvalues

def initialize_weights(L,
                       initialize_method='He'):

    if initialize_method=='He':
        nn.init.kaiming_normal_(L, nonlinearity='linear')

    if initialize_method=='Xavier':
        nn.init.xavier_normal_(L)

    if initialize_method is None:
        return(L)

def combine_pvalues_row(row,
                        method):
    p_values = [row['P.adj_Gene'], row['P.adj_Peak']]
    result = combine_pvalues(p_values, method).pvalue
    return result

def closure(G,
            P,
            L,
            optimizer,
            lambda_l2,
            losses):

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
            downsample=None):

    if downsample is not None:

        try:
            num = float(downsample)
        except ValueError:
            print("Pleses input the number of downsampling cells!")
            return

        if downsample > rna_adata.n_obs:
            downsample = rna_adata.n_obs

        print('Using ' + str(downsample) + ' cells to calculate linkage.')
        random_indices = np.random.choice(rna_adata.n_obs, size=downsample, replace=False)
        rna_adata = rna_adata[random_indices].copy()
        atac_adata = atac_adata[random_indices].copy()

    atac_adata.var.index = atac_adata.var.index.str.replace(':', '-')
    chrs = natsorted([element for element in np.unique(meta[3].tolist()) if element.startswith('chr')])
    linkage = []
    losses_list = []

    for chr in tqdm(chrs):

        chr_p = chr + '-'
        chr_gene = np.unique(
            meta.iloc[np.where((meta[3].to_numpy() == chr) & (meta[2].to_numpy() == 'Gene Expression'))][1].to_numpy())
        chr_peak = [item for item in atac_adata.var.index if chr_p in item]
        chr_gene = np.sort(np.array(list(set(chr_gene).intersection(set(rna_adata.var.index)))))
        chr_peak = np.sort(np.array(list(set(chr_peak).intersection(set(atac_adata.var.index)))))

        gene_to_index = {gene: idx for idx, gene in enumerate(rna_adata.var.index)}
        gene_indices = [gene_to_index[gene] for gene in chr_gene if gene in gene_to_index]
        chr_gene_exp_data = rna_adata.X[:, gene_indices].T.toarray()

        peak_to_index = {peak: idx for idx, peak in enumerate(atac_adata.var.index)}
        peak_indices = [peak_to_index[peak] for peak in chr_peak if peak in peak_to_index]
        chr_peak_exp_data = atac_adata.X[:, peak_indices].T.toarray()

        select_gene = chr_gene[np.where(np.sum(chr_gene_exp_data > 0, axis=1) > min_cell)[0]]
        select_peak = chr_peak[np.where(np.sum(chr_peak_exp_data > 0, axis=1) > min_cell)[0]]

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

        G = torch.from_numpy(chr_gene_exp_data).cuda(0)
        P = torch.from_numpy(chr_peak_exp_data).cuda(0)

        if normalize=='row':
            G = (G - torch.mean(G, dim=1, keepdim=True)) / torch.std(G, dim=1, keepdim=True)
            P = (P - torch.mean(P, dim=1, keepdim=True)) / torch.std(P, dim=1, keepdim=True)

        if normalize=='col':
            G = (G - torch.mean(G, dim=0)) / torch.std(G, dim=0)
            P = (P - torch.mean(P, dim=0)) / torch.std(P, dim=0)

        L = torch.rand((G.shape[0], P.shape[0]), requires_grad=True, device="cuda")
        initialize_weights(L, initialize_method=initialize_method)

        losses = []

        optimizer = LBFGS([L], lr=lr, max_iter=max_iter)
        closure_fn = closure(G, P, L, optimizer, lambda_l2, losses)
        optimizer.step(closure_fn)
        losses_list.append(losses)
        result = pd.DataFrame(L.cpu().detach().numpy(), index=select_gene, columns=select_peak)
        linkage.append(result)
        torch.cuda.empty_cache()
        gc.collect()

    if plot:
        plot_losses(losses_list, chrs)

    return linkage

def calculate_pvalue(l,
                     meta_data,
                     upstearm = 500000,
                     downstream = 500000,
                     n_jobs = 1,
                     method = 'pearson'):

    melted_chrs = []

    for l_chr in tqdm(l):

        df_reset = l_chr.reset_index()
        melted_chr = pd.melt(df_reset, id_vars=['index'])
        melted_chr.columns = ['Gene', 'Peak', 'Value']

        gene_p_values = []
        gene_index = []

        gene_to_indices = {}
        for idx, gene in enumerate(melted_chr['Gene']):
            if gene not in gene_to_indices:
                gene_to_indices[gene] = []
            gene_to_indices[gene].append(idx)

        for index, row in l_chr.iterrows():
            values = row.values
            median_value = np.median(values)
            mad_value = np.median(np.abs(values - median_value)) * 1.4826
            p_values = norm.sf(values, loc=median_value, scale=mad_value)
            adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            ind = gene_to_indices.get(index, [])
            gene_index.extend(ind)
            gene_p_values.extend(adjusted_p_values)

        peak_p_values = []
        for index, col in l_chr.items():
            values = col.values
            median_value = np.median(values)
            mad_value = np.median(np.abs(values - median_value)) * 1.4826
            p_values = norm.sf(values, loc=median_value, scale=mad_value)
            adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            peak_p_values.extend(adjusted_p_values)

        melted_chr.loc[gene_index, 'P.adj_Gene'] = gene_p_values
        melted_chr['P.adj_Peak'] = peak_p_values

        df_gene = meta_data.iloc[np.where(meta_data[1].isin(l_chr.index))[0], :]
        df_gene.columns = ['Ensemble', 'Gene', 'Type', 'Chromosome', 'Start', 'End']
        df_gene = df_gene.reset_index()

        df_peak = pd.DataFrame(l_chr.columns, columns=['Peak'])
        df_peak[['Chromosome', 'Start', 'End']] = df_peak['Peak'].str.split('-', n=2, expand=True)

        index_distances = []
        for index, row in df_gene.iterrows():
            gene = row.Gene
            range_gene = pr.PyRanges(df_gene.iloc[[index]])
            range_gene.End = range_gene.End + upstearm
            range_gene.Start = range_gene.Start - downstream
            range_peak = pr.PyRanges(df_peak)
            range_intersect_peak = range_peak.overlap(range_gene)
            if len(range_intersect_peak) > 0:
                intersect_peak = range_intersect_peak.Peak.tolist()
                distance_df = pd.DataFrame(
                    {"Gene": range_gene.Gene.tolist() * len(intersect_peak), "Peak": intersect_peak})
                linkage_gene_df = melted_chr.iloc[gene_to_indices[gene], :]
                index_distance = linkage_gene_df.index[np.where(linkage_gene_df['Peak'].isin(distance_df['Peak']))[0]]
                index_distances.extend(index_distance)

        melted_chr_distance = melted_chr.iloc[index_distances, :]
        melted_chr_distance = melted_chr_distance.drop_duplicates(subset=['Gene', 'Peak'])

        if n_jobs == 1:
            combine_p_values = []
            for index, row in melted_chr_distance.iterrows():
                new_value = combine_pvalues_row(row, method)
                combine_p_values.append(new_value)
            melted_chr_distance['Combine_p_value'] = combine_p_values

        if n_jobs > 1:
            melted_chr_distance['Combine_p_value'] = Parallel(n_jobs=n_jobs)(
                delayed(combine_pvalues_row)(row, method) for index, row in melted_chr_distance.iterrows())

        melted_chrs.append(melted_chr_distance)

    result = pd.concat(melted_chrs, ignore_index=True)

    return(result)




