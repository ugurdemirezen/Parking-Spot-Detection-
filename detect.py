import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

if __name__ == '__main__':
    # Load a model
    model = YOLO('C:/Users/Administrator/PycharmProjects/SensorRead/runs/detect/train28/weights/best.pt')
    VIDEO="result.mp4"
    colors = sv.ColorPalette.default()
    video_info = sv.VideoInfo.from_video_path(VIDEO)
    # extract video frame
    generator = sv.get_video_frames_generator(VIDEO)
    iterator = iter(generator)
    frame = next(iterator)
    cv2.imwrite("videokapak.jpg",frame)
    # initiate polygon zone
    polygons = [

        np.array([[86, 544], [47, 639], [62, 645], [97, 545]]),
        np.array([[102, 545], [55, 657], [107, 674], [153, 560]]),
        np.array([[157, 562], [107, 666], [138, 673], [178, 563]]),
        np.array([[194, 568], [150, 694], [209, 714], [262, 577]]),
        np.array([[277, 588], [227, 698], [276, 705], [320, 589]]),
        np.array([[336, 592], [289, 713], [335, 723], [365, 602]]),
        np.array([[370, 573], [328, 737], [408, 755], [457, 585]]),
        np.array([[472, 599], [420, 740], [492, 753], [534, 604]]),
        np.array([[552, 613], [510, 750], [585, 759], [619, 615]]),
        np.array([[634, 617], [609, 760], [674, 774], [688, 626]]),
        np.array([[710, 591], [665, 793], [800, 804], [817, 597]]),
        np.array([[844, 596], [821, 804], [932, 806], [935, 591]]),
        np.array([[949, 601], [941, 807], [1034, 801], [1048, 595]]),
        np.array([[1066, 615], [1070, 782], [1162, 786], [1149, 619]]),
        np.array([[1183, 639], [1203, 793], [1282, 787], [1252, 631]]),
        np.array([[1282, 618], [1311, 776], [1394, 776], [1346, 624]]),
        np.array([[1361, 589], [1388, 794], [1485, 789], [1450, 588]]),
        np.array([[1460, 568], [1499, 783], [1612, 770], [1554, 562]]),
        np.array([[1567, 599], [1601, 745], [1652, 731], [1615, 586]]),
        np.array([[1629, 594], [1673, 732], [1728, 728], [1670, 597]]),
        np.array([[1672, 559], [1745, 742], [1818, 723], [1749, 568]]),
        np.array([[1397, 434], [1412, 509], [1599, 501], [1583, 430]]),
        np.array([[1172, 438], [1184, 534], [1417, 522], [1395, 427]]),
        np.array([[938, 440], [934, 538], [1154, 530], [1149, 434]]),
        np.array([[729, 434], [726, 524], [923, 529], [919, 428]]),
        np.array([[519, 418], [505, 512], [692, 516], [698, 424]]),
        np.array([[366, 432], [349, 509], [492, 512], [495, 437]]),
        np.array([[251, 421], [235, 494], [336, 496], [352, 433]]),
        np.array([[163, 429], [152, 491], [235, 495], [240, 433]]),
        np.array([[98, 427], [76, 484], [150, 487], [159, 425]]),
        np.array([[49, 422], [48, 473], [88, 478], [92, 426]]),
        np.array([[0, 416], [5, 465], [45, 467], [56, 419]])
    ]
    colors = sv.ColorPalette.default()
    video_info = sv.VideoInfo.from_video_path(VIDEO)

    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=video_info.resolution_wh
        )
        for polygon
        in polygons
    ]
    zone_annotators = [ #polygone settings
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=4,
            text_thickness=5,
            text_scale=1
        )
        for index, zone
        in enumerate(zones)
    ]
    box_annotators = [#detection bounding box settings
        sv.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=1,
            text_thickness=0.1,
            text_scale=1,
        )
        for index
        in range(len(polygons))
    ]


    def process_frame(frame: np.ndarray, i) -> np.ndarray:

        # detection
        results = model.predict(frame)[0]
        detections = sv.Detections.from_ultralytics(results) # This method, accepts model results from both detection and segmentation models.
        detections = detections[(detections.class_id == 1) & (detections.confidence > 0.1)]

        sum_parked_car=0 #We defined this variable to find the number of parked vehicles.
        for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            frame = zone_annotator.annotate(scene=frame)
            print(len(detections_filtered))
            if len(detections_filtered)==1:
                sum_parked_car=sum_parked_car+1

        occupancy_rate = (sum_parked_car / len(polygons)) * 100
        print("Occupancy Rate:",occupancy_rate)
        return frame

    sv.process_video(source_path=VIDEO, target_path="result.mp4",
                     callback=process_frame)

