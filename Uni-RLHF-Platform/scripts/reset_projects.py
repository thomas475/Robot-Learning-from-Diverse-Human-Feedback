import pymysql
from create_table import cfg

try:
    connection = pymysql.connect(
        host=cfg['host'],
        user=cfg['username'],
        password=cfg['password'],
        database=cfg['database_name']
    )
    cursor = connection.cursor()
    # logger.Logger.info('[database] successfully connect database!')
except pymysql.err.OperationalError as e:
    # If the connection fails, determine whether to create a new database based on the error code
    if e.args[0] == 1049:  # Could not find database
        connection = pymysql.connect(
            host=cfg['host'],
            user=cfg['username'],
            password=cfg['password']
        )
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE {cfg['database_name']}")
        cursor.execute(f"USE {cfg['database_name']}")
        # logger.Logger.info(f"[database] Database {cfg['database_name']} created and connected")
    else:
        # If it is not because the database was not found, the error is thrown again
        raise e

sql1 = '''USE ''' + cfg['database_name'] + ''';'''
cursor.execute(sql1)

sql2 = '''DELETE FROM UserProject;'''
cursor.execute(sql2)

sql3 = '''DELETE FROM Project;'''
cursor.execute(sql3)