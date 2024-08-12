import pymysql
from create_table import cfg


def remove_projects(project_ids='all', project_statuses='all'):
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

    cursor.execute('''USE ''' + cfg['database_name'] + ''';''')
    cursor.execute('''SELECT project_id, project_name, status FROM Project;''')
    projects = cursor.fetchall()

    if project_ids == 'all':
        project_ids = {project[0] for project in projects}
    if project_statuses == 'all':
        project_statuses = {project[2] for project in projects}

    project_list = []
    for project_id, project_name, status in projects:
        if project_id in project_ids and status in project_statuses:
            project_list.append(project_name)
            cursor.execute("DELETE FROM Project WHERE project_id = %s", (project_id,))
            cursor.execute("DELETE FROM UserProject WHERE project_id = %s", (project_id,))
            cursor.execute("SELECT query_id FROM Query WHERE project_id = %s", (project_id,))
            query_ids = [row[0] for row in cursor.fetchall()]
            if query_ids:
                format_strings = ','.join(['%s'] * len(query_ids))
                cursor.execute(f"DELETE FROM QueryAnnotator WHERE query_id IN ({format_strings})", tuple(query_ids))
            cursor.execute("DELETE FROM Query WHERE project_id = %s", (project_id,))

    if input('Do you want to remove the following projects ? [Y/n]\n    ' + '\n    '.join(project_list) + '\n') == 'Y':
        connection.commit()

    connection.close()


if __name__ == "__main__":
    remove_projects(project_statuses=['creation'])