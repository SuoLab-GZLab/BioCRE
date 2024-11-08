def diopy_read(filename, assay_name = 'RNA'):
    """Read diopy converted Seurat data"""
    adata = diopy.input.read_h5(file = filename, assay_name = assay_name)
    return adata