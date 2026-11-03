# Campus Data Import Examples

**These files are illustrative examples, not surveyed IIT Hyderabad data.**

Coordinates here are placeholders demonstrating the ingestion pipeline's
formats and behavior. They are **not** verified GPS positions, and must
not be used for real navigation. This is a deliberate choice: fabricating
coordinates and presenting them as real would be actively dangerous for a
navigation-for-visually-impaired-users product if anyone trusted them.

## How to bring in real data

1. **OpenStreetMap** (recommended primary source for roads/paths/footprints):
   ```
   osmium export campus.osm.pbf -o campus.geojson
   ```
   then place `campus.geojson` in the directory pointed to by
   `Settings.campus_data_directory`. See
   `src/campus_nav/infrastructure/ingestion/geojson_importer.py` for the
   exact expected `properties` on each feature.

2. **Personal survey data** (walking the campus, noting names/coordinates):
   use `example_survey.csv` as a template — a spreadsheet export works
   directly.

3. **Accessibility notes / landmark write-ups**: use `example_note.md` as
   a template — one Markdown file per note.

4. **Hand-authored structured data** (richest format — nodes, edges, and
   standalone facts in one file): use `example_locations.yaml` as a
   template.

Every file dropped into the configured directory is picked up automatically
on next startup — no code changes required. See `docs/architecture.md`
for the full ingestion pipeline design.
