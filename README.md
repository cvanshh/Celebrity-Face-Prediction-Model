# Celebrity Look-Alike Finder

A Streamlit application that compares a user's uploaded face with
precomputed celebrity face embeddings and displays the closest match.

## Features

-   Upload a JPG, JPEG, or PNG image.
-   Generate a face embedding using DeepFace with the FaceNet model.
-   Compare the uploaded face embedding with entries stored in
    `embeddings.pkl` using cosine similarity.
-   Display the closest celebrity match, similarity score, a simple
    confidence label, and the top three matches.
-   Display a sample image from the matching celebrity's folder in
    `faces/`.

## Project structure

``` text
Celebrity Face Prediction/
├── app.py
├── embeddings.pkl
├── faces/
│   ├── Aamir_Khan/
│   ├── ...
│   └── <other celebrity folders>/
└── face testing/
    └── sample images for manual testing/
```

Keep `app.py`, `embeddings.pkl`, and the `faces/` directory together in
the same project folder. The app expects these paths relative to the
current working directory.

## Requirements

-   Python 3.10 or 3.11 is a sensible starting point for compatibility
    with the TensorFlow/DeepFace dependency stack.
-   Dependencies are listed in `requirements.txt`.
-   First-time setup downloads model weights used by DeepFace/FaceNet,
    so the first prediction can take longer and requires an internet
    connection.

## Installation

Open a terminal in the project folder.

### Windows PowerShell

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can run the
environment's Python directly:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the app

From the folder containing `app.py`, run:

``` powershell
python -m streamlit run app.py
```

Streamlit will print a local address, commonly `http://localhost:8501`,
which you can open in your browser.

## How to use

1.  Start the app.
2.  Upload a clear JPG, JPEG, or PNG photo containing a face.
3.  Wait while DeepFace generates an embedding and the app compares it
    with the saved database.
4.  Review the closest match, similarity score, confidence label, and
    top matches.

## Data and model notes

-   `embeddings.pkl` is a precomputed database of celebrity embeddings.
    The app does not regenerate this database during normal use.
-   The embeddings must be compatible with the FaceNet model and have
    the same embedding dimensions as those produced by
    `DeepFace.represent(model_name="Facenet")`.
-   Celebrity images in `faces/` are used to display a representative
    image for the predicted match.
-   The displayed score is cosine similarity, not a calibrated
    probability that the prediction is correct. The High/Medium/Low
    labels are simple thresholds in the code, not statistically
    validated confidence estimates.
-   Results depend on image quality, face detection, and the coverage
    and quality of the saved celebrity embeddings.

## Troubleshooting

### `ModuleNotFoundError`

Activate the project's virtual environment and install the packages
again:

``` powershell
python -m pip install -r requirements.txt
```

### DeepFace downloads weights or takes a long time on first run

This is expected on first use. Ensure the machine has internet access
and allow the download to finish.

### `embeddings.pkl` not found

Run Streamlit from the project folder that contains `embeddings.pkl`, or
update the paths in `app.py` to use paths relative to the script.

### No matching celebrity image appears

Check that the corresponding celebrity folder exists inside `faces/` and
contains supported image files.

### Face detection or image errors

Try a clear, well-lit, front-facing JPG or PNG image with one visible
face.




## Limitations and responsible use

This project is a demonstration of face-embedding similarity, not a
verified identity-recognition system. A look-alike match does not
establish identity, relationship, ethnicity, or any other personal
attribute. Avoid using its output for consequential decisions. Only use
images you have permission to process, and handle uploaded photos and
biometric embeddings carefully.




#LAST THING: IF YOU WANT MORE CONFIDENCE SCORE, ADD MORE IMAGES(FOLDERS) TO THE DATASET AKA Faces.rar
#DO ADD YOUR OWN DATASET 
