import os
import dlib
from PIL import Image, ImageDraw
import numpy as np

# file path
# dir paths
DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__))) # this file path
DIR_DB = os.path.join(DIR_PATH, 'database')
DIR_RES = os.path.join(DIR_DB, 'resources')
# DIR_PREDICTOR = os.path.join('D:\SCHOOL\AI2\Research Paper\Program\program_code\database/resources\predictor',
#                              'predictor_face_landmark.dat')
DIR_PREDICTOR = os.path.join(DIR_RES, 'predictor', 'predictor_face_landmark.dat')
DIR_EXPORT = os.path.join(DIR_DB, 'exports')

print(DIR_RES)
# image dir paths
DIR_FACE_IMAGES = os.path.join(DIR_RES, 'gt_db')
DIR_MASK_IMAGES = os.path.join(DIR_RES, 'mask_images')

class FaceOcclusionProcessing:
    def __init__(self, detector, predictor): # dir_face
        self.detector = detector
        self.predictor = predictor

    def process_crop_faces(self, dir_face, image_settings=(320, 0.75)):
        img_load = dlib.load_rgb_image(dir_face)

        detect_face = self.detector(img_load, 1) # 1 upsampling
        faces = dlib.full_object_detections()
        for detection in detect_face:
            faces.append(self.predictor(img_load, detection))
        img_size, img_pads = image_settings
        images = dlib.get_face_chips(img_load, faces,
                                     size=img_size, padding=img_pads)
        processed_img = [img for img in images]
        return processed_img[0]

    def occlude_lower_faces(self, face_img):
        # img_face = Image.fromarray(face_img)
        img_face = face_img
        detect = self.detector(img_face)
        shape = self.predictor(img_face, detect[0])

        lower_face_Y = max(shape.part(1).y, shape.part(16).y, shape.part(29).y)

        im = Image.fromarray(face_img)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, lower_face_Y, 300, 300], fill='black')

        return np.asarray(im)

    def occlude_except_eye_and_eyebrow(self, face_img):
        img_face = face_img
        detect = self.detector(img_face)
        shape = self.predictor(img_face, detect[0])

        top = max(shape.part(19).y, shape.part(24).y)
        bottom = shape.part(28).y
        left = shape.part(0).x
        right = shape.part(16).x
        sx, sy = 1.1, 1.2
        sy_offset = 2

        top = top - ((top * sy) - top) * sy_offset
        bottom = bottom * sy

        im = Image.fromarray(face_img)
        draw = ImageDraw.Draw(im)

        draw.rectangle([0, bottom, 300, 300], fill='black')
        draw.rectangle([right, 0, 300, 300], fill='black')
        draw.rectangle([left, 0, 0, 300], fill='black')
        draw.rectangle([0, top, 300, 0], fill='black')

        return np.asarray(im)

    def process_occlude_faces(self, face_img, dir_mask):
        self.face_img = face_img
        face_img = Image.fromarray(self.face_img)
        self.mask_img = Image.open(dir_mask)

        face_features = ('jaw', 'nose_line')
        face_landmarks = self.get_landmarks(face_features)

        for face_landmark in face_landmarks:
            nose_anchor, jaw_anchor = self.get_face_anchors(face_features, face_landmark)
            img, rect = self.transform_mask(nose_anchor, jaw_anchor)
            face_img.paste(img, rect, img)
        return np.asarray(face_img)

    def get_landmarks(self, features=('jaw', 'nose_line')):
        # load predictor and detector to create the face shapes
        detector = dlib.get_frontal_face_detector()
        # detect = detector(self.img_face, 1)
        detect = detector(self.face_img, 1)
        predictor = dlib.shape_predictor(DIR_PREDICTOR)
        shape = predictor(self.face_img, detect[0])

        # get the coordinates of nose, nose bridge and chin
        face_pts = np.empty([68, 2], dtype=int)
        for i in range(68): # shape predictor has 68
            face_pts[i][0] = shape.part(i).x # shape.part(i) gets the face pts
            face_pts[i][1] = shape.part(i).y
            # print(f'{face_pts[i]}, ', end='')

        # returns points of nose line and jaw
        return [{ features[0] : face_pts[:17], features[1] : face_pts[27: 31]}]

    def get_face_anchors(self, face_features, face_landmark: dict):
        # face_feature[1] = nose, [...][1] = second point from above as the anchor point
        nose_anchor = np.array( face_landmark[face_features[1]][1] )
        jaw_points = face_landmark[face_features[0]]
        jaw_anchor_center = np.array(jaw_points[(len(jaw_points) // 2)]) # get midpoint of the jaw points

        # nose, jaw(left, center, right)
        return nose_anchor, (jaw_points[2], jaw_anchor_center, jaw_points[-3])

    def transform_mask(self, nose_anchor, jaw_anchor):
        jaw_anchor_left, jaw_anchor_center, jaw_anchor_right = jaw_anchor

        # get the width and height of mask image
        w, h = self.mask_img.width, self.mask_img.height

        # height of nose to center jaw
        occlusion_height = int(np.linalg.norm(nose_anchor - jaw_anchor_center))
        side_mask_lr = ((jaw_anchor_left, (0, 0, w // 2, h)),
                        (jaw_anchor_right, (w // 2, 0, w, h)))

        mask_parts = []
        for i in range(2):
            img = self.mask_img.crop(side_mask_lr[i][1])
            # get the distance of line from nose line to chin and cheek
            dist_w = np.abs(np.cross(jaw_anchor_center - nose_anchor, nose_anchor - side_mask_lr[i][0]) /
                            np.linalg.norm(jaw_anchor_center - nose_anchor))
            mask_parts.append(img.resize((int(dist_w), occlusion_height)))
            # print(f'mp: {mask_parts[i]}')

        # merge mask
        mask_l, mask_r = mask_parts
        mask_img = Image.new('RGBA', (mask_l.width + mask_r.width, occlusion_height))
        mask_img.paste(mask_l, (0, 0), mask_l)
        mask_img.paste(mask_r, (mask_l.width, 0), mask_r)

        # transformation
        mask_coord = (jaw_anchor_left[0], nose_anchor[1])

        return mask_img, mask_coord



def main():

    FOP = FaceOcclusionProcessing(detector=dlib.get_frontal_face_detector(),
                                  predictor=dlib.shape_predictor(DIR_PREDICTOR))
    mask_names, mask_path, mask_images = get_mask_images()

    image_settings = (256, 0.25)

    for root, dir, files in os.walk(DIR_FACE_IMAGES):
        if len(files) > 0:
            subject_name = os.path.join(root[-3:])
            mk_data_path = os.path.join(DIR_EXPORT, 'database', subject_name)
            for fimg in files:
                i_cnt = int(fimg[:-4])
                res_img_path = os.path.join(root, fimg)
                dir_crop = os.path.join(DIR_EXPORT, 'cropped_faces', subject_name)
                print_status(mode=0, str_input=(subject_name, fimg, res_img_path))

                cimg = FOP.process_crop_faces(dir_face=res_img_path, image_settings=image_settings)
                save_stat = save_image(cimg, dir_crop, fimg)
                print_status(mode=1, str_input=save_stat)

                # for database listing
                # if i_cnt < 6 or i_cnt >= 12:
                #     save_image(cimg, mk_data_path, fimg)

                m_cnt = i_cnt % 5
                for m in range(3):
                    dir_occluded = os.path.join(DIR_EXPORT, mask_names[m], subject_name)
                    oimg = FOP.process_occlude_faces(face_img=cimg,
                                                     dir_mask=os.path.join(mask_path[m],mask_images[m][m_cnt]))
                    print_status(mode=2, str_input=(mask_names[m], mask_images[m][m_cnt]))
                    save_stat = save_image(oimg, dir_occluded, fimg)
                    print_status(mode=1, str_input=save_stat)

                    # for database listing
                    # if i_cnt in (6, 7) and m == 0: save_image(oimg, mk_data_path, fimg)
                    # if i_cnt in (8, 9) and m == 1: save_image(oimg, mk_data_path, fimg)
                    # if i_cnt in (10, 11) and m == 2: save_image(oimg, mk_data_path, fimg)

                ofimg = FOP.occlude_lower_faces(face_img=cimg)
                print_status(mode=2, str_input= ('Lower Face', 'Black'))
                save_stat = save_image(ofimg, os.path.join(DIR_EXPORT, 'occluded_faces', subject_name), fimg)
                print_status(mode=1, str_input=save_stat)

                oeeimg = FOP.occlude_except_eye_and_eyebrow(face_img=cimg)
                print_status(mode=2, str_input=('Eye and', 'Eyebrow'))
                save_stat = save_image(oeeimg, os.path.join(DIR_EXPORT, 'eye_brow_only_faces', subject_name), fimg)
                print_status(mode=1, str_input=save_stat)
                # # for database listing
                # save_image(ofimg)

            # print_status(mode=3, str_input=(subject_name, mk_data_path))

def print_status(mode, str_input):
    str_in = [inp for inp in str_input]
    switch = {
        0:f'\nPROCESSING BATCH ({str_in[0]}): '
          f'\n    Cropping      :  {str_in[0]} - {str_in[1]}',
        1:f'    Saved at      :  {str_input}\n',
        2:f'    Occluded with :  {str_in[0]} - {str_in[1]}',
        3:f'\nBUILDING DATA BASE ({str_in[0]}): '
          f'\n    Saved at      : {str_in[1]}',
    }
    print(switch.get(mode, 'Invalid'))

def get_mask_images():
    rt, fl = [], []
    for root, dir, files in os.walk(DIR_MASK_IMAGES):
        if len(files) > 0:
            rt.append(root)
            fl.append(files)
    n = [rtn[len(DIR_MASK_IMAGES) + 1:] for rtn in rt]
    return n, rt, fl

def save_image(image, folder_name, img_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    path = os.path.join(folder_name, img_name)
    dlib.save_image(image, path)
    return path


if __name__ == '__main__':
    main()

