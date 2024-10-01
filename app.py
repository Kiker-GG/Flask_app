from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import time
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Length

# Экземпляр приложения
app = Flask(__name__)

# Настройки для подключения к базе данных
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///types.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret_key_here'

# Инициализация базы данных
db = SQLAlchemy(app)

class UserForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=20)])
    age = IntegerField('Age', validators=[DataRequired()])
    city = StringField('City', validators=[DataRequired(), Length(min=2, max=20)])
    submit = SubmitField('Submit')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    city = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'User({self.name}, {self.age}, {self.city})'
    
class Machine_learning_type(db.Model):
    name = db.Column(db.String(80), primary_key=True)
    accuracy = db.Column(db.Float, nullable=False)
    time = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

# Маршрут для главной страницы
@app.route('/', methods=['GET', 'POST'])
def home():
    global data
    greeting = ""
    plot_div = ""   
    form = UserForm()
    if request.method == 'POST':
        if 'button_iris' in request.form:
            return redirect(url_for('iris'))
        elif 'button_wine' in request.form:
            return redirect(url_for('wine'))
        elif 'button_res' in request.form:
            return redirect(url_for('res'))
    
        if form.validate_on_submit():
            name = form.name.data
            age = form.age.data
            city = form.city.data

            greeting = f"Hello, {name} from {city}!"

            #Добавление данных в базу данных
            new_user = User(name=name, age=age, city=city)
            db.session.add(new_user)
            db.session.commit()

            # Извлекаем все данные из базы данных для построения данных
            users = User.query.all()
            data = [{'Name': user.name, 'Age': user.age, 'City': user.city} for user in users]

            # Создаём график распределения возраста
            fig = px.histogram(data, x='Age', title='Age Distribution')
            plot_div = pio.to_html(fig, full_html=False)

            return render_template('index.html',form=form, greeting=greeting, plot_div=plot_div)
        

    return render_template('index.html', form=form)

# Определяем маршрут для страницы с датасетом ирисов
@app.route('/iris', methods=['GET', 'POST'])
def iris():
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target

    plot_div = ""
    accuracy_results = ""

    # Строим графики на основе данных
    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=iris_data.feature_names,
        color='species',
        title='Iris Dataset Scatter Matrix',
        labels={col: col.replace(" (cm)", "") for col in iris_data.feature_names}
    )
    plot_div = pio.to_html(scatter_matrix, full_html=False)#, default_height='700px')

    if request.method == 'POST':
        if "back" in request.form:
            return redirect(url_for('home'))

        # Разделение данных
        (X_train, X_test, y_train, y_test) = train_test_split(iris_data.data, iris_data.target, train_size=0.3, random_state=42)

        # Стандартизация
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Обучение моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Support Vector Machines": SVC(kernel='rbf', gamma=0.1, C=1.0),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        }

        accuracy_results = {}
        for name, model in models.items():
            start = time.time()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_results[name] = round(accuracy_score(y_test, y_pred), 2)

            end = time.time() - start

            new_model = Machine_learning_type(name=name, accuracy=accuracy_results[name], time=end)
            db.session.merge(new_model)
            db.session.commit()

        # Создаём столбчатую диаграмму с результатами
        bar_chart = go.Figure([go.Bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()))])
        bar_chart.update_layout(title="Model Accuracy Comparsion", xaxis_title="Model", yaxis_title="Accuracy")
        plot_div += pio.to_html(bar_chart, full_html=False)

    return render_template('iris.html', plot_div=plot_div, accuracy_results=accuracy_results)

@app.route('/wine', methods=['GET', 'POST'])
def wine():
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['species'] = wine_data.target

    html_table = df.head(15).to_html(classes=['table'])

    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=['alcohol','color_intensity','hue','od280/od315_of_diluted_wines'],
        color='species'
    )
    plot_div = pio.to_html(scatter_matrix, full_html=False, default_height='1000px')

    if request.method == 'POST':
        if "back" in request.form:
            return redirect(url_for('home'))
        
        plot_div = ""
        arr = [0.3, 0.5, 0.7]
        for t in arr:
            (X_train, X_test, y_train, y_test) = train_test_split(wine_data.data, wine_data.target, train_size=t, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Обучение моделей
            models = {
                "KNN": KNeighborsClassifier,
                "Decision Tree": DecisionTreeClassifier,
                "Gradient Boosting": GradientBoostingClassifier
            }

            accuracy_results = {}
            for name, model in models.items():
                if t == 0.3:
                    start = time.time()
                m = model()
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                accuracy_results[name] = round(accuracy_score(y_test, y_pred), 2)
                if t == 0.3:
                    end = time.time() - start

                    new_model = Machine_learning_type(name=name, accuracy=accuracy_results[name], time=end)
                    db.session.merge(new_model)
                    db.session.commit()

            bar_chart = go.Figure([go.Bar(x=list(accuracy_results.keys()), y=list(accuracy_results.values()))])
            bar_chart.update_layout(title="Model Accuracy Comparsion with train size = " + str(t), xaxis_title="Model", yaxis_title="Accuracy")
            
            buttons = [
                dict(label='Bar', method='update', args=[{'type': 'bar'}]),
                dict(label='Scatter', method='update', args=[{'type': 'scatter'}])
            ]

            bar_chart.update_layout(
                updatemenus=[
                    dict(
                        buttons=buttons,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.5,
                        xanchor="left",
                        y=1.1,
                        yanchor="top"
                    )
                ]
            )

            
            plot_div += pio.to_html(bar_chart, full_html=False)

        return render_template('wine.html', plot_div=plot_div)


    return render_template('wine.html', html_table=html_table, plot_div=plot_div)

@app.route('/results', methods=['GET','POST'])
def res():
    models_of_ML = Machine_learning_type.query.all()
    data = [{'Name': m.name, 'Accuracy': m.accuracy, 'Time': m.time} for m in models_of_ML]

    if request.method == 'POST':
        if 'button_sort_by_accurasy_ascending' in request.form:
            data.sort(key = lambda x: x['Accuracy'])
        elif 'button_sort_by_accurasy_descending' in request.form:
            data.sort(key = lambda x: x['Accuracy'], reverse=True)
        elif "back" in request.form:
            return redirect(url_for('home'))

    html_table = pd.DataFrame(data).to_html(classes=['table'])

    return render_template('res.html', html_table=html_table)

# Запуск
if __name__ == '__main__':
    app.run(debug=True)


