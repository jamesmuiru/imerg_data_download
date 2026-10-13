"""Example usage in Google Colab after installing dependencies and uploading scripts."""
from imerg_wrapper import get_imerg

result = get_imerg(
    start_date="2025-01-01",
    end_date="2025-01-03",
    bbox=(36.0, -1.5, 38.0, 0.5),  # west, south, east, north
    output_dir="/content/IMERG",
    dataset_name="GPM_3IMERGHH",
    convert_to_zarr=True,
    max_workers=4,
)

print(result)
