from typing import List, Optional

from .. import SessionItem, get, register_session_item


def _set_genes(x: Optional[List[str]]) -> None:
    dataloader = get("dataloader")
    if dataloader and x and dataloader.dataset.genes != x:
        dataloader.dataset.genes = x
        dataloader.reset_workers()


register_session_item(
    "genes", SessionItem(setter=_set_genes, default=None, persistent=True)
)
