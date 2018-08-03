from __future__ import print_function

import glob
import json
import os
import sys
import uuid
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from flask import Flask, request, redirect, render_template, Response ,jsonify
from tensorflow.python.platform import gfile
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from db import Base, File
import datetime
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

sys.path.append(os.getcwd())

app = Flask(__name__)
engine = create_engine('sqlite:///example.db')
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = 'static\\images'
app.config['UPLOAD_ORIGINAL_FOLDER'] = 'static\\original'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('/aicontainer.html')

@app.route('/demo')
def demo():
    return render_template('/demo.html')

@app.route('/savebox',methods=['GET', 'POST'])
def savebox():
    jsondata = request.get_json()
    filename = jsondata['filename']
    filepath = jsondata['filepath']
    saveboxvalue = str(jsondata['saveboxvalue'])
    new_file = File(id = filename , Value = saveboxvalue , root = filepath ,CreateTime = datetime.datetime.now() )
    session.add(new_file)   
    session.commit()
    return jsonify(True)

@app.route('/encoding', methods=['GET', 'POST'])
def upload_face_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # container recognition
        if file and allowed_file(file.filename):
            # delete file
            images = glob.glob(app.config['UPLOAD_FOLDER'] + '\\*')
            for item in images:
                os.remove(item)

            # save file
            path = os.path.join(app.config['UPLOAD_FOLDER'] , file.filename)
            path = os.path.join(app.config['UPLOAD_ORIGINAL_FOLDER'] , file.filename)
            file.save(path)

            # rename file
            filename, extension = os.path.splitext(path)
            filename = str(uuid.uuid1()) + extension
            os.rename(path, os.path.join(app.config['UPLOAD_ORIGINAL_FOLDER'] , filename))    

        return jsonify(os.path.join(app.config['UPLOAD_ORIGINAL_FOLDER'] , filename))

    # If no valid image file was uploaded, show the file upload form:
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    # get data
    jsonData = request.get_json()
    ori_file = jsonData['path']

    # init session
    cfg_from_file('ctpn/text.yml')
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    im_names = glob.glob(os.path.join(ori_file))
    for im_name in im_names:
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        im_dict = draw_boxes(img, im_name, boxes, scale)

    return Response(json.dumps(im_dict), mimetype='application/json')


@app.route('/dailydata', methods=['GET'])
def dailydata():
    todaydata = []
    for data in session.query(File).filter(File.CreateTime >= datetime.date.today()).all():
        result = {
        "id": data.id,
        "root": data.root,
        "Value": data.Value,
        "CreateTime": data.CreateTime
        }
        todaydata.append(result)
    session.close()
    return jsonify(todaydata)


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    im_dict = dict()
    base_name = image_name.split('\\')[-1]
    with open('static\\images\\' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for index, box in enumerate(boxes):
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

            # crop image
            file_name = base_name.split('.')
            image_crop = cv2.imread(image_name)[min_y:max_y, min_x:max_x]

            # resize image
            r = 100.0 / image_crop.shape[0]
            width, height = (int(image_crop.shape[1] * r), 100)
            img_resize = cv2.resize(image_crop, (width, height))

            # save image
            whitelist = "01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ."
            save_name = file_name[0] + '_' + str(index) + '.' + file_name[1]
            cv2.imwrite(os.path.join("static\\images", save_name), img_resize)
            im_text = pytesseract.image_to_string(image=img_resize,
                                                  lang='cntr',
                                                  config="-psm 6 -c tessedit_char_whitelist=" + whitelist)

            im_dict['static\\images\\' + save_name] = im_text

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("static\\images", base_name), img)
    return im_dict





if __name__ == "__main__":
    app.run(debug=True)
