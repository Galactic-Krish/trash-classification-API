from fastai.vision import load_learner, open_image, defaults, Path
import torch

defaults.device = torch.device('cpu')
learner = load_learner(Path(''))

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files['image']
    img_file = file.stream

    img = open_image(img_file)
    pred_class, pred_idx, outputs = learner.predict(img)
    formatted_outputs = ["{:.1f}".format(value) for value in
                         [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(zip(learner.data.classes, map(str, formatted_outputs)),
                        key=lambda p: p[1],
                        reverse=True
                        )

    stats = int(open('num-classifications.txt', 'r').read())

    num_classifications = open('num-classifications.txt', 'w')
    num_classifications.write(str(stats + 1))

    num_classifications.close()

    return jsonify({pred_probs[0][0]: float(pred_probs[0][1]), pred_probs[1][0]: float(pred_probs[1][1]), pred_probs[2][0]: float(pred_probs[2][1])})


@app.route("/stats", methods=["GET"])
def stats():

    return "{} images classified!".format(open('num-classifications.txt', 'r').read())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
