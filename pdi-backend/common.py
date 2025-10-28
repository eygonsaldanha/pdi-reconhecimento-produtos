import datetime
import hashlib

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_hash(txt_base='txt'):
    return hashlib.sha256(f'{txt_base}{datetime.datetime.now()}'.encode('utf-8')).hexdigest()[:25]
