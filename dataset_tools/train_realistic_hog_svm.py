import os
import random

import cv2
import numpy as np

DATASET = os.path.expanduser("~/ros2_ws/realistic_sim_hog_dataset")
OUT_MODEL = os.path.expanduser("~/ros2_ws/realistic_sim_hog_svm.yml")

CLASSES = {
    "cup": 0,
    "bowl": 1,
    "coke_can": 2,
    "cordless_drill": 3,
    "hammer": 4,
    "background": 5,
}

WINDOW = (128, 128)
MAX_BACKGROUND = 200
TEST_FRACTION = 0.15

random.seed(11)

hog = cv2.HOGDescriptor(WINDOW, (16, 16), (8, 8), (8, 8), 9)


def load_samples():
    samples = []
    labels = []

    for class_name, class_id in CLASSES.items():
        folder = os.path.join(DATASET, class_name)
        files = sorted(os.listdir(folder))

        if class_name == "background" and len(files) > MAX_BACKGROUND:
            random.shuffle(files)
            files = files[:MAX_BACKGROUND]

        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[:2] != WINDOW[::-1]:
                img = cv2.resize(img, WINDOW)
            features = hog.compute(img).flatten()
            samples.append(features)
            labels.append(class_id)

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    X, y = load_samples()
    print("Total samples:", len(y))
    for name, cid in CLASSES.items():
        print(f"  {name}: {(y == cid).sum()}")

    idx = np.arange(len(y))
    rng = np.random.default_rng(11)
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_test = int(len(y) * TEST_FRACTION)
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    print(f"Train: {len(y_train)}  Test: {len(y_test)}")

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(2.5)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    svm.save(OUT_MODEL)
    print("Saved:", OUT_MODEL)

    _, pred = svm.predict(X_test)
    pred = pred.flatten().astype(np.int32)

    correct = (pred == y_test).sum()
    print(f"Test accuracy: {correct}/{len(y_test)} = {correct / len(y_test):.3f}")

    id_to_name = {v: k for k, v in CLASSES.items()}
    n_classes = len(CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_test, pred):
        confusion[t, p] += 1

    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{id_to_name[i][:6]:>7s}" for i in range(n_classes)))
    for i in range(n_classes):
        print(f"{id_to_name[i][:7]:>7s} " + " ".join(f"{confusion[i,j]:7d}" for j in range(n_classes)))

    print()
    print("Per-class precision/recall:")
    for i in range(n_classes):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {id_to_name[i]:16s} precision={precision:.3f} recall={recall:.3f} support={confusion[i,:].sum()}")


if __name__ == "__main__":
    main()
