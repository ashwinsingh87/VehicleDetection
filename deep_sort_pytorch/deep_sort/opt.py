class Opt:
    def __init__(self, source = 0, save_vid = True, save_txt = False, upper = 250, lower = 175, motorcycle = 60, car = 100, truck = 80, bus = 80, distance = 100, track = False, bbox = False):
        self.yolo_weights = 'yolov5/yolov5s.pt'
        self.deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.output = 'inference/output'
        self.img_size = 640
        self.conf_thres = 0.1
        self.iou_thres = 0.5
        self.fourcc = 'mp4v'
        self.device = 0
        self.show_vid = True
        self.classes = [2, 3, 5, 7]
        self.agnostic_nms = False
        self.augment = False
        self.evaluate = False
        self.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'
        self.upper = upper
        self.lower = lower
        self.motorcycle = motorcycle
        self.car = car
        self.truck = truck
        self.bus = bus
        self.distance = distance
        self.save_txt = save_txt
        self.save_vid = save_vid
        self.source = source
        self.track = track
        self.bbox = bbox

class LicenceOpt():
    def __init__(self, model, img, im0):
        self.model = model
        self.img = img
        self.im0 = im0
        self.device = 'cpu'
        self.imgsz = [640, 640]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.line_thickness = 1
        self.half = False