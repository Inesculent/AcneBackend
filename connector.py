import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv
import os
import psycopg2
# Load environment variables from .env file
load_dotenv()
pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,  # Number of connections in the pool
        pool_reset_session=True,
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE"),
        port=3306
    )


# THESE ARE NOT USED DIRECTLY BY THE APPLICATION

# The following are test queries for the database


def get_cursor():

    if pool is not None:

        connection = pool.get_connection()
        cursor = connection.cursor()
        return cursor, connection

    return "Database failed to initialize"

def print_table_structure(table_name):
    try:
        # Get a connection from the pool
        cursor, connection = get_cursor()

        # Execute DESCRIBE query
        query = f"DESCRIBE {table_name};"
        cursor.execute(query)

        # Fetch and display results
        columns = cursor.fetchall()
        print(f"Structure of table '{table_name}':")
        print(f"{'Field':<20}{'Type':<20}{'Null':<10}{'Key':<10}{'Default':<20}{'Extra':<10}")
        print("-" * 90)

        for col in columns:
            print(f"{col[0]:<20}{col[1]:<20}{col[2]:<10}{col[3]:<10}{str(col[4]):<20}{col[5]:<10}")

    finally:
        # Close the connection
        if connection.is_connected():
            connection.close()

def get_inputs(user_id):

    cursor, connection = get_cursor()

    with open("sq/num_inputs.sql", 'r') as f:
        cmd = f.read()

    cmd += user_id

    print(cmd)
    #cursor.execute(command)

    cursor.close()
    connection.close()


def authenticate_user(user_id):


    cursor, connection = get_cursor()

    query = "SELECT EXISTS( SELECT 1 FROM users WHERE user_id = %s) AS user_exists"
    cursor.execute(query, user_id,)

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    print(result[0])

    return result[0] == 1

def print_users():
    cursor, connection = get_cursor()
    query = "SELECT * FROM users"
    cursor.execute(query)
    columns = cursor.fetchall()
    print(columns)
    cursor.close()
    connection.close()



def add_user(user_id, name, email):
    cursor, connection = get_cursor()

    query = "INSERT INTO users (user_id, name, email) VALUES (%s, %s, %s)"
    cursor.execute(query, (user_id, name, email,))

    connection.commit()

    print(f"User {user_id} : {name} added successfully")


    cursor.close()
    connection.close()

def remove_user(user_id):
    cursor, connection = get_cursor()
    query = "DELETE FROM users WHERE user_id = %s"
    cursor.execute(query, (user_id, ))
    connection.commit()
    print(f"User {user_id} removed successfully")
    cursor.close()
    connection.close()


def add_data(acne_cells, acne_coverage, date, user_id, file_name):

    cursor, connection = get_cursor()

    query = "INSERT INTO user_data (acne_cells, acne_coverage, date, user_id, filename) VALUES (%s, %s, %s, %s, %s)"

    cursor.execute(query, (acne_cells, acne_coverage, date, user_id, file_name,))
    connection.commit()
    print(f"Data {user_id} : {file_name} added successfully")
    cursor.close()
    connection.close()


def delete_data(user):

    cursor, connection = get_cursor()

    query = "DELETE FROM user_data WHERE user_id = %s"

    cursor.execute(query, (user,))
    connection.commit()
    print(f"Data {user} removed successfully")
    cursor.close()
    connection.close()



def get_data(user_id, limit):
    cursor, connection = get_cursor()

    query = "SELECT * FROM user_data WHERE user_id = %s LIMIT %s"
    cursor.execute(query, (user_id, limit,))

    results = cursor.fetchall()

    cursor.close()
    connection.close()


    return results

def alter_table():

    cursor, connection = get_cursor()
    query = "ALTER TABLE user_data MODIFY COLUMN date DATETIME"
    cursor.execute(query)
    connection.commit()
    cursor.close()
    connection.close()

    print("Good")




# Test connection
#if __name__ == '__main__':

    #print(delete_data('HzdS3jLyV8hAQfAfMNPJuXpq9LE2'))


