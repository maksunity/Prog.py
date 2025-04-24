import psycopg2

try:
    conn = psycopg2.connect(
        dbname="your_database",
        user="your_user",
        password="your_password",
        host="localhost",
        port="5432"
    )
    print("Connection successful")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
