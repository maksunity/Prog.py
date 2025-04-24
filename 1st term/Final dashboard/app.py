from flask import Flask, render_template, request, redirect, session, url_for, flash
import psycopg2
from psycopg2.extras import RealDictCursor
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.secret_key = "mylittlesecret"  # Замените на свой
bcrypt = Bcrypt(app)

conn = psycopg2.connect(
    dbname="db",
    user="dashboard_user",
    password="password",
    host="localhost",
    port="5342"
)


# Функция для выполнения запросов
def execute_query(query, params=(), fetchone=False, fetchall=False):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        if fetchone:
            return cur.fetchone()
        if fetchall:
            return cur.fetchall()
        conn.commit()


# Роут для главной страницы (страница входа)
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = execute_query(
            "SELECT * FROM users WHERE username = %s",
            (username,),
            fetchone=True
        )

        if user and bcrypt.check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))

        flash("Неверный логин или пароль", "danger")
    return render_template("login.html")


# Роут для регистрации
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

        try:
            execute_query(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hashed_password)
            )
            flash("Регистрация успешна! Войдите в аккаунт.", "success")
            return redirect(url_for("login"))
        except psycopg2.IntegrityError:
            conn.rollback()
            flash("Имя пользователя уже занято", "danger")
    return render_template("register.html")


# Роут для личного кабинета
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        pulse = request.form["pulse"]
        blood_pressure = request.form["blood_pressure"]

        execute_query(
            "INSERT INTO stats (user_id, pulse, blood_pressure) VALUES (%s, %s, %s)",
            (session["user_id"], pulse, blood_pressure)
        )
        flash("Данные добавлены", "success")

    stats = execute_query(
        "SELECT * FROM stats WHERE user_id = %s ORDER BY timestamp DESC",
        (session["user_id"],),
        fetchall=True
    )
    return render_template("dashboard.html", stats=stats)


# Роут для выхода
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
