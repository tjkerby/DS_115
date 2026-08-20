# DS 115 Day 2 integration

Copy these files into the root of the DS 115 Quarto repository.

## What changes

- Adds `slides/day2.qmd`.
- Adds `slides/day2_assets/` with synthetic data, figures, generated Markdown snippets, and the Python regeneration script.
- Adds the Day 2 link to `index.qmd`.
- Adds `requirements-day2.txt` for optional asset regeneration.

## What does not change

Day 2 inherits the existing settings from `slides/_metadata.yml` and the existing course design from `theme.scss`. No changes to `_quarto.yml`, `slides/_metadata.yml`, or `theme.scss` are required.

## Publishing

The slide deck is static at Quarto render time. Python is not required to build the website because the tables and figures are committed as generated assets.

To regenerate the Day 2 data and assets after editing the simulation:

```bash
pip install -r requirements-day2.txt
python slides/day2_assets/day2_student_success_data.py
```

Then render or preview the Quarto website as usual.
