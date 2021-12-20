import cv2
import os
import json
from operator import itemgetter

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' +  directory)

def processing_annotations(json_path):
    with open(json_path, 'r', encoding='utf8') as json_f:
        json_data = json.load(json_f)

        categories = json_data["categories"]

        annotations = json_data["annotations"]
        sort_annotations = sorted(annotations, key=itemgetter("frame"))
        process_annotations = []
        for annotation in sort_annotations:
            if annotation["id"][0] == "p" or annotation["id"][0] == "a":
                annotation["gender"] = [category["gender"] for category in categories if
                                        category["id"] == annotation["id"]][0]
                annotation["age"] = [category["age"] for category in categories if
                                     category["id"] == annotation["id"]][0]
                process_annotations.append(annotation)

    return process_annotations

def extract_frame(video_path, process_annotations, file_name, train_f):
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    checked_idx = 0
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            if ret:
                split_process_annotations = process_annotations[checked_idx:]
                for process_annotation in split_process_annotations:
                    if process_annotation["frame"] > frame_no:
                        break
                    bbox = process_annotation["bbox"]
                    crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()
                    jpg_file_name = file_name + "_" + process_annotation["id"] + "_" + str(process_annotation["frame"]) + ".jpg"
                    cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                    gender_w = {"male": "1 0 ", "female": "0 1 "}.get(process_annotation["gender"])
                    age_w = {"child": "1 0 0 0 ", "teenager": "0 1 0 0 ", "adult": "0 0 1 0 ", "senior": "0 0 0 1 "}.get(process_annotation["age"])
                    top_type_w = {"long_sleeve": "1 0 0 0 ", "short_sleeve": "0 1 0 0 ", "sleeveless": "0 0 1 0 ", "onepice": "0 0 0 1 "}.get(process_annotation["top_type"])
                    top_color_w = {"red": "1 0 0 0 0 0 0 0 0 0 0 ", "orange": "0 1 0 0 0 0 0 0 0 0 0 ",
                                   "yellow": "0 0 1 0 0 0 0 0 0 0 0 ", "green": "0 0 0 1 0 0 0 0 0 0 0 ",
                                   "blue": "0 0 0 0 1 0 0 0 0 0 0 ", "purple": "0 0 0 0 0 1 0 0 0 0 0 ",
                                   "pink": "0 0 0 0 0 0 1 0 0 0 0 ", "brown": "0 0 0 0 0 0 0 1 0 0 0 ",
                                   "white": "0 0 0 0 0 0 0 0 1 0 0 ", "grey": "0 0 0 0 0 0 0 0 0 1 0 ",
                                   "black": "0 0 0 0 0 0 0 0 0 0 1 "}.get(process_annotation["top_color"])
                    bottom_type_w = {"long_pants": "1 0 0 0 ", "short_pants": "0 1 0 0 ", "skirt": "0 0 1 0 ", "none": "0 0 0 1 "}.get(process_annotation["bottom_type"])
                    bottom_color_w = {"red": "1 0 0 0 0 0 0 0 0 0 0 ", "orange": "0 1 0 0 0 0 0 0 0 0 0 ",
                                      "yellow": "0 0 1 0 0 0 0 0 0 0 0 ", "green": "0 0 0 1 0 0 0 0 0 0 0 ",
                                      "blue": "0 0 0 0 1 0 0 0 0 0 0 ", "purple": "0 0 0 0 0 1 0 0 0 0 0 ",
                                      "pink": "0 0 0 0 0 0 1 0 0 0 0 ", "brown": "0 0 0 0 0 0 0 1 0 0 0 ",
                                      "white": "0 0 0 0 0 0 0 0 1 0 0 ", "grey": "0 0 0 0 0 0 0 0 0 1 0 ",
                                      "black": "0 0 0 0 0 0 0 0 0 0 1 ", "none": "0 0 0 0 0 0 0 0 0 0 0 ", }.get(process_annotation["bottom_color"])
                    accessories_w = {"carrier": "1 0 0 0 0 0 ", "umbrella": "0 1 0 0 0 0 ",
                                     "bag": "0 0 1 0 0 0 ", "hat": "0 0 0 1 0 0 ", "glasses": "0 0 0 0 1 0 ",
                                     "none": "0 0 0 0 0 1 "}.get(process_annotation["accessories"])
                    pet_w = str(process_annotation["pet"])

                    dataset_line = jpg_file_name + " " + gender_w + age_w + top_type_w + top_color_w + bottom_type_w + bottom_color_w + accessories_w + pet_w + "\n"
                    cv2.imwrite('processed_images/' + jpg_file_name, crop_img)
                    train_f.write(dataset_line)
                    checked_idx += 1
                frame_no += 1
            else:
                break
    else:
        print("can't open video.")
    cap.release()
    cv2.destroyAllWindows()

def processing(video_dir, train_f):
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            file_name, extension = os.path.splitext(file)
            if extension == ".mp4":
                print(file)
                json_path = root + "/" + file_name + ".json"
                if os.path.exists(json_path):
                    process_annotations = processing_annotations(json_path)
                else:
                    with open("./error.txt", 'w', encoding='utf8') as error_f:
                        error_f.write(root + "/" + file + "\n")
                    print("Error : not matched " + root + "/" + file)
                    continue

                video_path = root + "/" + file
                extract_frame(video_path, process_annotations, file_name, train_f)

if __name__ == "__main__":
    createFolder("./processed_images")

    video_dir = "../sample_video/video"

    with open("./dataset.txt", 'w', encoding='utf8') as train_f:
        processing(video_dir, train_f)