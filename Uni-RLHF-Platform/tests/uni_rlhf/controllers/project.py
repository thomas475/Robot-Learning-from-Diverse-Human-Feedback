from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import time

app = Flask(__name__)
db = SQLAlchemy()

def to_dict(model_instance):
    return {c.name: getattr(model_instance, c.name) for c in model_instance.__table__.columns}

class Project(db.Model):
    __tablename__ = 'Project'
    project_id = db.Column(db.String(80), primary_key=True)
    project_name = db.Column(db.String(120))
    description = db.Column(db.Text)
    due_time = db.Column(db.DateTime)
    visibility = db.Column(db.Integer)
    dataset_path = db.Column(db.String(200))
    domain = db.Column(db.String(80))
    task = db.Column(db.String(80))
    environment_name = db.Column(db.String(80))
    instruction = db.Column(db.Text)
    mode = db.Column(db.String(80))
    sampler_type = db.Column(db.String(80))
    feedback_type = db.Column(db.String(80))
    query_num = db.Column(db.Integer)
    query_length = db.Column(db.Integer)
    video_width = db.Column(db.Integer)
    video_height = db.Column(db.Integer)
    fps = db.Column(db.Integer)
    creator = db.Column(db.Integer, db.ForeignKey('User.user_id'))
    create_time = db.Column(db.DateTime)
    status = db.Column(db.String(80))
    annotation_num = db.Column(db.Integer)
    question = db.Column(db.Text)
    is_deleted = db.Column(db.Boolean, default=False)

def generate_video(cfg, project_id):
    context = {}
    exec(f"from uni_rlhf.datasets import {cfg['mode']}_{cfg['domain'].lower()} as dataset_module", context)
    datasets = context['dataset_module']
    with app.app_context():
        dataset = datasets.Dataset(project_id=cfg['project_id'], domain=cfg['domain'], task=cfg['task'],
                                   environment_name=cfg['environment_name'], mode=cfg['mode'],
                                   sampler_type=cfg['sampler_type'], feedback_type=cfg['feedback_type'],
                                   query_num=cfg['query_num'], query_length=cfg['query_length'],
                                   fps=cfg['fps'], video_width=cfg['video_width'], video_height=cfg['video_height'],
                                   save_dir=cfg['save_dir'])
        video_info_list, video_url_list, query_id_list = dataset.generate_video_resources()

# domain = 'd4rl'
# task = 'mujoco'
# environment_name = 'hopper-medium-v2'
#
# domain = 'd4rl'
# task = 'antmaze'
# environment_name = 'antmaze-umaze-v2'

domain = 'd4rl'
task = 'kitchen'
environment_name = 'kitchen-complete-v0'

test_project = Project(
    project_id=str(int(time.time())),
    project_name='Test Project',
    description='Test description',
    due_time=datetime.utcnow(),
    visibility=1,
    dataset_path='',
    domain=domain,
    task=task,
    environment_name=environment_name,
    instruction='<p>Enter the project instruction for annotators.</p>',
    mode='offline',
    sampler_type='random',
    feedback_type='evaluative',
    query_num=1,
    query_length=200,
    video_width=200,
    video_height=200,
    fps=30,
    creator=1,
    create_time=datetime.utcnow(),
    status='active',
    annotation_num=0,
    question='test_question'
)

# cfg = {key: value for key, value in test_project.__dict__.items()}
cfg = to_dict(test_project)
cfg['save_dir'] = f"{'./uni_rlhf/vue_part/src/assets/video/'}{test_project.project_id}_{test_project.environment_name}_{test_project.mode}_{test_project.feedback_type}"

generate_video(cfg, test_project.project_id)

print('Video stored in ' + cfg['save_dir'])

import os
os.system('xdg-open "%s"' % cfg['save_dir'])