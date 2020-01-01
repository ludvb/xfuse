from typing import List, Optional

from .. import SessionItem, get, register_session_item


def _set_genes(x: Optional[List[str]]) -> None:
    dataloader = get("dataloader")
    if dataloader and x:
        for slide in dataloader.dataset.data.slides.values():
            slide.data.genes = x


register_session_item("genes", SessionItem(setter=_set_genes, default=None))
