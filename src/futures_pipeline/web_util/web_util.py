from pathlib import Path
from datetime import datetime
import dominate
from dominate.tags import div, img, body, h1, link, p, a
from ..config import WEB_PATH
from bs4 import BeautifulSoup, Tag
from typing import Optional, cast


""" TODO: Make a context_lengths page and context_length_metrics page, 
creating buttons for each resolution that directs to the pages. 
with each call to tune_context_length, parse the context_lengths page, 
find the div with id="resolutions", search if the button already exists,
if exists, update the resolutions page, if not, create page.
"""
'''Build the scaffolding for the web pages.'''
def web_scaffolding():
    if WEB_PATH.exists():
        return

    hyperparams_dir = WEB_PATH / "hyperparams"
    zeroshot_dir = hyperparams_dir / "zeroshot"
    finetuned_dir = hyperparams_dir / "finetuned"

    zeroshot_dir.mkdir(parents=True)
    finetuned_dir.mkdir(parents=True)

    main_page = WEB_PATH / "main.html"
    hyperparams_page = hyperparams_dir / 'hyperparams.html'
    zeroshot_page = zeroshot_dir / 'nav.html'
    finetuned_page = finetuned_dir / 'nav.html'

    for page in [main_page, hyperparams_page, zeroshot_page, finetuned_page]:
        page.touch()

    main_doc: dominate.document = dominate.document(title="main")

    with main_doc:
        with div(id="buttons"):
            with div(id="hyperparams"):
                a("hyperparameter optimization", href="hyperparams/hyperparams.html")
    with open(main_page, 'w') as f:
        f.write(str(main_doc))

    hyperparams_doc = dominate.document(title="hyperparams")
    with hyperparams_doc:
        with div(id="buttons"):
            a("Back", href=main_page.resolve())
            a("Zero Shot", href=WEB_PATH/f"hyperparams/zeroshot/nav.html")
            a("Fine Tuned", href=WEB_PATH/f"hyperparams/finetuned/nav.html")
            a(href=main_page.resolve())

    with open(hyperparams_page, 'w') as f:
        f.write(str(hyperparams_doc))

    for page in [zeroshot_page, finetuned_page]:
        doc = dominate.document(title=page.stem)
        with doc:
            with div(id="buttons"):
                a("Back", href=hyperparams_page)
                a("Context Length Predictions", href="context_lengths.html")
                a("Context Length Metrics", href="context_length_metrics.html")

        with open(page, 'w') as f:
            f.write(str(doc))


def display_figs_on_page(figs: list[tuple[Path, list[str]]], page_path: Path):
    page_path.touch(exist_ok=True)
    last_update = datetime.today()
    doc: dominate.document = dominate.document(title="foo")
    with doc.head:
        link(rel="stylesheet", href="../../src/output.css")
    with doc:
        with div():
            with a("Back", href="../context_lengths.html"): ...
        with div(f"last_update: {str(last_update)}"): ...
        with div(id='images', cls="flex flex-wrap"):
            for fig, text in figs:
                with div(id=f"{fig.stem}-cnt", cls="flex flex-col items-center"):
                    with div(id=fig.stem):
                        img(src=fig.resolve())
                    if len(text):
                        with div(id=f"{fig.name}-text"):
                            for item in text:
                                p(item)

    with open(page_path, 'w') as f:
        f.write(str(doc))


"""
(1) Create page if it doesn't exist.
(2) Parse for resolutions div
(3) seaach for resolution
"""
def update_resolution_page(page: Path, resolution: str, href_to: Path):
    def make_page():
        if not page.exists():
            page.touch()
        doc = dominate.document()
        with doc:
            with div():
                with a("Back", href="nav.html"): ...
            with div(id="resolutions"): ...
        with open(page, 'w') as fp:
            fp.write(str(doc))

    if not page.exists():
        make_page()

    with open(page, 'r') as fp:
        soup = BeautifulSoup(fp)

    resolutions_div: Optional[Tag] = soup.find('div', id="resolutions")

    if resolutions_div is None:
        make_page()
        with open(page, 'r') as fp:
            soup = BeautifulSoup(fp)

    resolutions_div = soup.find('div', id="resolutions")

    if resolutions_div is None:
        raise ValueError("Could not find the resolution div.")

    tag = resolutions_div.find(id=resolution)

    if tag is None:
        tag = resolutions_div.append(soup.new_tag("div", id=resolution))
        tag = tag.append(soup.new_tag("a", string=resolution, href=str(href_to.resolve()))) # type: ignore

    with open(page, 'w') as fp:
        fp.write(soup.prettify())


    



