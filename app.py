from flask import Flask, render_template
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import Required, AnyOf
from flask_navigation import Navigation

app = Flask(__name__)
nav = Navigation(app)
app.config['SECRET_KEY'] = 'reallyreallyreallyreallysecretkey'

manager   = Manager(app)
bootstrap = Bootstrap(app)
moment    = Moment(app)

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('main.html')

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    return render_template('calculator.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    return render_template('data.html')

if __name__ == '__main__':
    app.run()