import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import sklearn.cluster
#base
def parse_annotations(annotation_dir, image_dir, normalize=False):
    annotations = [os.path.join(os.path.abspath(annotation_dir), f) for f in os.listdir(annotation_dir)
                   if f.lower().endswith(".xml")]

    result = []
    for annotation in tqdm(annotations):
        #print("annotations")
        #print(annotations)
        root = ET.parse(annotation).getroot()
        img_path = os.path.join(image_dir, root.find("filename").text)

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        img_objects = []
        objects = root.findall("object")
        for object in objects:
            name = object.find("name").text
            bndbox = object.find("bndbox")
            x1 = int(bndbox.find("xmin").text)
            y1 = int(bndbox.find("ymin").text)
            x2 = int(bndbox.find("xmax").text)
            y2 = int(bndbox.find("ymax").text)
            
            if normalize:
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
            img_objects.append((x1, y1, x2, y2, name))

        result.append((img_path, img_objects))
    return result

def run_kmeans(data, num_anchors, tolerate, verbose=False):#tolerate離開迭代的最小誤差,verbose : 是否输出详细信息
    km = sklearn.cluster.KMeans(n_clusters=num_anchors, tol=tolerate, verbose=verbose)
    km.fit(data)
    return km.cluster_centers_
#base

#v2
def generate_anchors(params):
    num_anchors = int(params["num_anchors"])
    image_dir = params["image_dir"]
    annotation_dir = params["annotation_dir"]
    tolerate = float(params["tolerate"])
    stride = int(params["stride"])
    input_w = int(params["input_w"])
    input_h = int(params["input_h"])

    annotations = parse_annotations(annotation_dir, image_dir, normalize=True)
    print("{} annotations found.".format(len(annotations)))
    class_names = set()
    data = []
    for annotation in annotations:
        obj = annotation[1]
        for o in obj:
            
            w = float(o[2] - o[0])
            h = float(o[3] - o[1])
            data.append([w, h])
            class_names.add(o[-1])
    
    anchors = run_kmeans(data, num_anchors, tolerate)
    #print("--------")
    #print(anchors)
    #print("--------")
    anchors = [[a[0] * input_w / stride, a[1] * input_h / stride] for a in anchors]
    #print("***************")
    #print(anchors)
    #print("***************")
    #anchors = np.reshape(anchors, [-1])


    return anchors, class_names
#v2




params = {'stride': '32', 'tolerate': '0.05', 'annotation_dir': '/home/jiemin/anchorGenerate/voc2007/', 'input_h': '416', 'image_dir': '/home/jiemin/anchorGenerate/voc2007/', 'num_anchors': '5', 'input_w': '416'}
num_anchors = int(params["num_anchors"])
generate_anchors = generate_anchors

anchors, class_names = generate_anchors(params)

print("anchor_X")
for index in range(0,num_anchors):
    print(anchors[index][0], end='')
    print(",", end='')

print("")

print("anchor_Y")
for index in range(0,num_anchors):
    print(anchors[index][0], end='')
    print(",", end='')

print("")


#print("Anchors: ")
#print("\t{}".format(anchors))
print("Class names: ")
print("\t{}".format(class_names))
