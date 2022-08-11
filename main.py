import cv2
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import numpy as np

#DIMENSÕES
WIDTH, HEIGHT = (1024, 768)

#CORES DAS CAIXAS
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#CLASSES (COCO - PRÉ DEFINIDAS):
CLASS_NAMES = []
with open("coco.names", "r") as f:
    CLASS_NAMES = [cname.strip() for cname in f.readlines()]

#FOTO (IMPORTAÇÃO)
imgx = tf.io.read_file("c:/Users/estef/Downloads/MLproject/images/000000122166.jpg")
imgx = tf.io.decode_image(imgx)
imgx = tf.image.resize(imgx, (HEIGHT, WIDTH))
imgxs = tf.expand_dims(imgx, axis=0)/255

#MODELO
model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3), 
    num_classes=80, 
    anchors=YOLOV4_ANCHORS,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5
    )
model.load_weights('yolov4.h5')

boxes, scores, classes, detections = model.predict(imgxs)

boxes = boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
scores = scores[0]
classes = classes[0].astype(int)
detections = detections[0]

imgx = np.asarray(imgx)
imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)

#DETECÇÕES (opencv-cv2)
for (classid, score, box) in zip(classes, scores, boxes):
    #COR DA CLASSE:
    color = COLORS[int(classid)%len(COLORS)]
    #NOME DA CLASSE:
    nameclass = f"{CLASS_NAMES[classid]} : {score}"
    #BOX DA DETECÇÃO:
    cv2.rectangle(imgx, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.putText(imgx, nameclass, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #VISUALIZAÇÃO DO RESULTADO:
    cv2.imwrite("detection.jpg", imgx)
