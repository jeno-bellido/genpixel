from flask import Flask, render_template, request
from numpy.random import randn
from keras.models import load_model
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import urllib
import base64

app = Flask(__name__)

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def plot_generated(examples, i):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.axis('off')
    axis.imshow(examples[i, :, :])
    return fig

@app.route('/generate', methods=['POST'])
def generate():
    n = request.form.get('n', type=int)
    model = load_model('generator_model_050.h5')
    latent_points = generate_latent_points(100, n)
    X = model.predict(latent_points)
    X = (X + 1) / 2.0
    images = []
    for i in range(n):
        fig = plot_generated(X, i)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        images.append(urllib.parse.quote(base64.b64encode(output.getvalue())))
    return {'images': images}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
