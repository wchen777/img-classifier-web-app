from flask import Flask, render_template, request
from predict import *


# create flask app
app = Flask(__name__, template_folder='./public')

# initialize model
model = init_model()


# home endpoint, render html file
@app.route('/')
def render():
    return render_template('index.html')


# submit button endpoint
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    img_data = request.get_data()
    process_image(img_data)
    print("predicting with model...")
    return str(predict(model))


# run app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
