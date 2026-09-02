"""Machine-readable transcription of AdaCLIP's main paper tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


INDUSTRIAL = {
    "MVTec AD": ((89.2, 90.6), (88.7, 43.4)),
    "VisA": ((85.8, 83.1), (95.5, 37.7)),
    "MPDD": ((76.0, 82.5), (96.1, 34.9)),
    "BTAD": ((88.6, 88.2), (92.1, 51.7)),
    "KSDD": ((97.1, 90.7), (97.7, 54.5)),
    "DAGM": ((99.1, 97.5), (91.5, 57.5)),
    "DTD-Synthetic": ((95.5, 94.7), (97.9, 71.6)),
}

MEDICAL_IMAGE = {
    "HeadCT": (91.4, 85.2),
    "BrainMRI": (94.8, 91.2),
    "Br35H": (97.7, 92.4),
}

MEDICAL_PIXEL = {
    "ISIC": (89.3, 71.4),
    "ColonDB": (90.4, 58.2),
    "ClinicDB": (84.4, 58.2),
    "TN3K": (77.2, 41.9),
}


def rows():
    output = []
    for dataset, (image, pixel) in INDUSTRIAL.items():
        output.append({"domain": "industrial", "dataset": dataset,
                       "i_auroc": image[0], "i_f1max": image[1],
                       "p_auroc": pixel[0], "p_f1max": pixel[1]})
    output.append({"domain": "industrial", "dataset": "Paper average",
                   "i_auroc": 90.2, "i_f1max": 89.6,
                   "p_auroc": 94.2, "p_f1max": 50.2})
    for dataset, image in MEDICAL_IMAGE.items():
        output.append({"domain": "medical", "dataset": dataset,
                       "i_auroc": image[0], "i_f1max": image[1],
                       "p_auroc": "", "p_f1max": ""})
    output.append({"domain": "medical", "dataset": "Image paper average",
                   "i_auroc": 94.6, "i_f1max": 89.6,
                   "p_auroc": "", "p_f1max": ""})
    for dataset, pixel in MEDICAL_PIXEL.items():
        output.append({"domain": "medical", "dataset": dataset,
                       "i_auroc": "", "i_f1max": "",
                       "p_auroc": pixel[0], "p_f1max": pixel[1]})
    output.append({"domain": "medical", "dataset": "Pixel paper average",
                   "i_auroc": "", "i_f1max": "",
                   "p_auroc": 85.3, "p_f1max": 57.4})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = rows()
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print(args.output)


if __name__ == "__main__":
    main()
