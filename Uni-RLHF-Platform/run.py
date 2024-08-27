import os
import time
import multiprocessing
import json

"""
    function for start celery
"""
def run_celery():
    os.system("echo Starting celery ...")
    # os.system("celery -A uni_rlhf.controllers.auth.celery worker -l INFO")
    os.system("celery -A uni_rlhf.controllers.auth.celery worker -l INFO -P solo")

"""
    function for start Flask server
"""
def run_flask_server(base_url, flask_port):
    os.system(f"echo Starting Flask server on http://{base_url}:{flask_port}/ ...")
    os.system(f"python -m uni_rlhf.controllers.auth --port {flask_port}")
    
"""
    function for start vue server
"""
def run_vue_server(base_url, vue_port):
    time.sleep(8)
    os.system("echo Starting vue server ...")
    os.system(f"npm run serve --prefix ./uni_rlhf/vue_part/ -- --host {base_url} --port {vue_port}")


if __name__ == '__main__':
    """
        Run Backend Server & Frontend Server at the same time.
    """
    with open(os.path.join(os.getcwd(), 'uni_rlhf', 'config.json'), 'r') as file:
        config = json.load(file)

    base_url = config['baseUrl']
    flask_port = config['flaskPort']
    vue_port = config['vuePort']
    
    p1 = multiprocessing.Process(target=run_celery)
    p2 = multiprocessing.Process(target=run_flask_server, args=(base_url, flask_port,))
    p3 = multiprocessing.Process(target=run_vue_server, args=(base_url, vue_port,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()