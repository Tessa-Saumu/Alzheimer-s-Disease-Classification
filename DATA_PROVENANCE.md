# Data usage and evidence boundary

The original image collection's source URLs, redistribution licence, patient identifiers and mapping from source scans to augmented slices have not been established. Existing files do not grant reuse permission. This repair adds no medical images and makes no clinical claim.

The executable audit accepts an explicit CSV with `path,split` columns and train/validation/test labels. Paths are relative to that CSV. Supply only images you have permission to process. The report stores the manifest SHA256, each image SHA256 and cross-split matching groups. Preserve that manifest and report together. Average-hash matches are investigation candidates: distinct flat images can collide, transformations may escape detection, and neither hash method identifies patients.

The synthetic automated check creates its own images in a temporary folder. It verifies audit mechanics, not the original 44,000-image collection or the original consumed split. Historical notebook/screenshot percentages remain historical observations and are not reproduced by this check.
