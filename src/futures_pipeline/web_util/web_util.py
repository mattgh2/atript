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

    main_page = WEB_PATH / "main.html"

    zeroshot_dir = WEB_PATH / "zeroshot"
    finetuned_dir = WEB_PATH / "finetuned"

    for dir in [zeroshot_dir, finetuned_dir]:
        hyperparam_dir = dir / f"hyperparameter_tuning"
        abalation_dir = dir / f"abalation"
        hyperparam_dir.mkdir(parents=True, exist_ok=True)
        abalation_dir.mkdir(parents=True, exist_ok=True)



    zeroshot_page = zeroshot_dir / 'nav.html'
    finetuned_page = finetuned_dir / 'nav.html'
    # abalation_page = WEB_PATH / f"abalation"
    # hyperparams_page = hyperparams_dir / 'hyperparams.html'


    for page in [main_page, zeroshot_page, finetuned_page]:
        page.touch()

    main_doc: dominate.document = dominate.document(title="main")

    with main_doc:
        with div(id="buttons"):
            with div(id="anchors"):
                a("Zero Shot", href=zeroshot_page)
                a("Fine Tuned", href=finetuned_page)

    with open(main_page, 'w') as f:
        f.write(str(main_doc))
    

    for dir, page in [(zeroshot_dir, zeroshot_page), (finetuned_dir, finetuned_page)]:
        doc = dominate.document(title=page.stem)
        with doc:
            with div(id="buttons"):
                a("Back", href=main_page)
                a('hyperparameter tuning', href="hyperparameter_tuning/hyperparameters.html")
                a('Feature Abalation', href="abalation/abalation.html")

        with open(page, 'w') as f:
            f.write(str(doc))

        hyperparam_doc = dominate.document(title='hyperparams')
        hyperparam_page = dir / f'hyperparameter_tuning/hyperparameters.html'

        with hyperparam_doc:
            with div(id='buttons'):
                a("Back", href="../nav.html")
                a("context_lengths", href="context_lengths.html")
                a("metrics", href="context_length_metrics.html")

        with open(hyperparam_page, 'w') as f:
            f.write(str(hyperparam_doc))
        

"""
Writes a sequence of figures to html.

@param fig_path
@param page_path the pathname where the html file is located.
"""
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
def update_resolution_page(page: Path, resolution: str, href_to: Path, back: Path):
    def make_page():
        if not page.exists():
            page.touch()
        doc = dominate.document()
        with doc:
            with div():
                with a("Back", href=back): ...
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
