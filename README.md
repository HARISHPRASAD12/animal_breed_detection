# Cattle Detection and Health Screening

A computer vision project that locates and counts visible cattle in photographs. The Streamlit interface accepts JPG, PNG and WebP uploads, displays cattle bounding boxes and scores, and exports annotated images and CSV records.

## Run on Windows

From the project folder in PowerShell, use the existing virtual environment or create one with `py -3.13 -m venv .venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Open the local address shown in the terminal. Use a photo of one animal, a herd, or a scene with no cattle. The threshold slider defaults to 0.35. The count reflects detections at the selected threshold. Once dependencies and model are present, analysis runs locally on CPU.

### Model setup

The application expects `models/object_detection_yolox_2022nov.onnx`. If it is missing, download the [official OpenCV Zoo YOLOX-S model](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx) into that path. The expected SHA-256 is:

```text
c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
```

## Architecture and methodology

1. Decode the upload and apply its camera orientation metadata.
2. Resize the RGB image to fit inside 640 × 640, preserve the aspect ratio, and pad the remaining area.
3. Run the COCO-pretrained YOLOX-S ONNX model with OpenCV DNN. Decode its candidate boxes and retain predictions whose highest-scoring category is COCO's cow class.
4. Multiply objectness by class score, filter by the selected threshold, and use non-maximum suppression at 0.45 IoU to remove overlapping proposals.
5. Map the remaining boxes to the original image coordinates, count them, and render the results.

[src/app.py](src/app.py) contains the upload and presentation flow, [src/detector.py](src/detector.py) contains inference and annotation, and [src/evaluate.py](src/evaluate.py) evaluates against human-labeled boxes.

## Health screening

Select a detected animal and review its crop. Record visible skin marks or wounds, swelling, and eye or nasal discharge. Choose **Unsure or obscured** when the relevant area cannot be assessed. Save observations before exporting the screening CSV. Photo screening covers visible signs and cannot establish overall health or diagnose disease. [Visual signs of cattle illness](https://extension.msstate.edu/publications/identifying-sick-or-injured-cattle)

### Trained breed and lumpy skin disease predictions

Each detected animal is cropped and classified by an ImageNet MobileNetV2 feature extractor with two supervised classification heads. Results appear beside the selected animal and in the downloadable CSV. The disease head distinguishes the dataset labels `healthycows` and `lumpycows`. Its negative result cannot establish general health. Breed estimates always select among the five training breeds.

To reproduce training, install `requirements-health.txt` and run `python src/train_predictions.py` from the project environment. Training uses `data/cow-lumpy.zip` and `data/cattle-breeds.zip`. [Disease photos referenced by both notebooks](https://www.kaggle.com/datasets/shivamagarwal29/cow-lumpy-disease-dataset) · [Additional breed-labeled photos](https://www.kaggle.com/datasets/anandkumarsahu09/cattle-breeds-dataset)

Training uses a seeded stratified 80/20 image split after removing duplicate resized images and filenames ending in `_c`. See `models/cattle_predictions.json` for class ordering, counts and confusion matrices (rows are reference labels). Animal identities and farm groupings were unavailable. The test images evaluate classification separately from cattle detection, and scores on new farms may differ.

### Optional additional skin abnormality model

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-health.txt
```

Automatic skin screening requires a model trained on labeled cattle photos. The photo review and observation export are available before training.

Prepare RGB cattle crops that resemble the boxes produced by the detector. Create `train`, `val`, and `test` directories, each containing `no_visible_abnormality` and `visible_abnormality` subdirectories. Have the labels reviewed by a qualified annotator, exclude unclear crops, and group photos from the same animal or capture session into one split. The trainer rejects byte-identical files across splits or conflicting labels within a split.

```powershell
.\.venv\Scripts\python.exe src/train_health.py path\to\labeled-crops --epochs 12
```

The MobileNetV2 trainer downloads ImageNet weights on first use, reports held-out metrics, and saves `models/cattle_skin_screening.keras`. The reported classification metrics use a decision threshold of 0.50. The app calls scores greater than 0.40 and less than 0.60 inconclusive. Measure the inconclusive rate and performance at the app thresholds on representative photos before relying on the results.


## Reproduce an evaluation

Collect photos of cattle and cattle-free scenes. Include challenging negatives such as buffalo or horses, plus varied breeds, viewpoints, lighting, and herd sizes. Keep images from the same farm, animal, or capture sequence together when splitting data. Choose a score threshold on validation images and use the test split for the final measurement.

Create a JSON manifest next to the evaluation images. Coordinates are pixel values in the EXIF-corrected image, in left, top, right, bottom order. The values below illustrate the format and should be replaced by your manual annotations.

```json
[
  {"image": "cattle.jpg", "boxes": [[30, 25, 415, 310]]},
  {"image": "empty-field.jpg", "boxes": []}
]
```

```powershell
.\.venv\Scripts\python.exe src/evaluate.py path\to\manifest.json --threshold 0.35
```

The evaluator matches predictions in score order to unused annotations at IoU ≥ 0.50 and reports precision, recall, and raw error counts. An undefined metric is reported as JSON null. Report the dataset composition and threshold alongside any results. This baseline uses pretrained weights, and detection scores are model scores. Small, occluded, distant and unusual-looking animals can reduce detection quality.

## Smoke test results

The Streamlit test runner exercised a real cattle upload, threshold changes, the PNG and CSV download controls, a blank upload, and a corrupt upload. At the default threshold, the detector found one animal in each of four photos and missed cattle in two. A representative, independently labeled test set is needed to estimate field performance. [Smoke test image source](https://github.com/srkamd/yolov8-cow-segmentation)

## References

- [YOLOX paper](https://arxiv.org/abs/2107.08430)
- [OpenCV Zoo implementation and model license](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)
- [COCO dataset](https://cocodataset.org/)

## Open on a phone

Connect the phone and computer to the same trusted Wi-Fi network. From the project directory, start the server with the command below, then open the Network URL printed in the terminal on the phone. Uploaded photos are processed on the computer.

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py --server.address 0.0.0.0
```

Choose **Choose file** to select a saved photo. **Use camera** works when the app is served over HTTPS and the browser has camera permission. For a local Wi-Fi HTTP address, take the photo with the phone camera first, then select the saved file.
