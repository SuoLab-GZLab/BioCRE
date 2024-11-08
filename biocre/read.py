import diopy
import scanpy

def scanpy_read_mtx(dirname,
                    var_names='gene_symbols',
                    make_unique=True,
                    prefix=None):

    """Read cellranger filter dir"""
    adata=scanpy.read_10x_mtx(dirname,
                              var_names=var_names,
                              make_unique=make_unique,
                              prefix=prefix,
                              cache=True,
                              gex_only=False)

    rna_adata = adata[:, adata.var['feature_types']=='Gene Expression']
    atac_adata = adata[:, adata.var['feature_types']=='Peaks']

    return rna_adata, atac_adata

def scanpy_read_h5(filename,
                   genome=None,
                   backup_url=None):

    """Read cellranger h5 file"""
    adata = scanpy.read_10x_h5(filename,
                               genome=genome,
                               backup_url=backup_url,
                               gex_only=False)

    rna_adata = adata[:, adata.var['feature_types'] == 'Gene Expression']
    atac_adata = adata[:, adata.var['feature_types'] == 'Peaks']

    return rna_adata, atac_adata

def diopy_read(filename,
               assay_name = 'RNA'):

    """Read diopy converted Seurat data"""
    adata = diopy.input.read_h5(file = filename,
                                assay_name = assay_name)
    return adata

