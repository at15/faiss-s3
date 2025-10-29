import os
import faiss

from dataclasses import dataclass


@dataclass
class IVFIndexMetadata:
    total_size: int
    n_clusters: int
    cluster_data_offset: int


def generate_meta_from_file(index_file_path: str) -> IVFIndexMetadata:
    file_size = os.path.getsize(index_file_path)
    # TODO: Should we use mmap or SKIP_IVF_DATA?
    index = faiss.read_index(index_file_path, faiss.IO_FLAG_MMAP)
    return _generate_meta(index, file_size)


# Generate metadata data base on write_index implementation in faiss
# Only works for ArrayInvertedLists (the default).
def _generate_meta(ivf_index, file_size: int) -> IVFIndexMetadata:
    # TODO: Support other IVF index types
    if not isinstance(ivf_index, faiss.IndexIVFFlat):
        raise ValueError("ivf_index must be a faiss.IndexIVFFlat")

    # Get the inverted lists (ArrayInvertedLists)
    invlists = ivf_index.invlists
    nlist = ivf_index.nlist
    code_size = invlists.code_size

    cluster_data_size = 0
    for i in range(nlist):
        list_size = ivf_index.get_list_size(i)
        # TODO: verify empty cluster does not have place holders
        cluster_data_size += list_size * code_size
        cluster_data_size += list_size * 8  # ids (sizeof(idx_t) = 8)

    cluster_data_offset = file_size - cluster_data_size

    return IVFIndexMetadata(
        total_size=file_size, n_clusters=nlist, cluster_data_offset=cluster_data_offset
    )
