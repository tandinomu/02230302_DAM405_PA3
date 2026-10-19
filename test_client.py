"""
Simulate traffic to the running service:
  1. 100 normal requests from the test set     -> INFO predictions, no drift
  2. 4 bad requests                             -> ERROR, HTTP 400
  3. 1 out-of-range request                     -> WARNING
  4. 100 drifted requests (size features x1.5,
     texture +5)                                -> DRIFT WARNING
  5. final stats

Start the service first (python app.py), then:  python test_client.py
"""
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

URL = "http://127.0.0.1:5001"


def post(body, show=False):
    r = requests.post(f"{URL}/predict", json=body)
    if show:
        print(r.status_code, r.json())
    return r


def main():
    data = load_breast_cancer(as_frame=True)
    _, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                            stratify=data.target, random_state=42)
    X = X_test.reset_index(drop=True)
    requests.post(f"{URL}/drift/reset")

    print("--- 1. normal traffic (100 requests) ---")
    correct = malignant = 0
    for i in range(100):
        row = X.iloc[i % len(X)]
        r = post({"features": row.tolist()}, show=i < 3)
        correct += r.json()["prediction"] == y_test.iloc[i % len(X)]
        malignant += r.json()["prediction"] == 0
    print(f"live accuracy on normal traffic: {correct}/100, "
          f"predicted malignant: {malignant}/100")
    print("drift check:", {k: v for k, v in
                           requests.get(f"{URL}/drift").json().items() if k != "details"})

    print("\n--- 2. bad requests (expect 400) ---")
    post({"features": [1, 2, 3]}, show=True)                 # wrong length
    post({"wrong": []}, show=True)                           # missing key
    post({"features": ["abc"] * 30}, show=True)              # not numbers
    post({"features": {"mean radius": 10}}, show=True)       # incomplete dict

    print("\n--- 3. out-of-range input (expect WARNING) ---")
    odd = X.iloc[0].copy()
    odd["mean area"] = 10_000
    post({"features": odd.tolist()}, show=True)

    print("\n--- 4. drifted traffic (100 requests) ---")
    requests.post(f"{URL}/drift/reset")
    size_cols = [c for c in X.columns if any(k in c for k in ("radius", "perimeter", "area"))]
    tex_cols = ["mean texture", "worst texture"]
    malignant = 0
    for i in range(100):
        row = X.iloc[i % len(X)].copy()
        row[size_cols] *= 1.5      # tumours look 50% larger
        row[tex_cols] += 5         # texture readings shifted (e.g. new scanner)
        malignant += post({"features": row.tolist()}).json()["prediction"] == 0
    print(f"predicted malignant under drift: {malignant}/100 "
          "(same patients, shifted measurements)")
    d = requests.get(f"{URL}/drift").json()
    print("drift check:", {k: v for k, v in d.items() if k != "details"})
    for f in d.get("details", [])[:5]:
        print("   ", f)

    print("\n--- 5. monitoring summary ---")
    print(requests.get(f"{URL}/stats").json())
    print("\nDone - see logs/service.log")


if __name__ == "__main__":
    main()
