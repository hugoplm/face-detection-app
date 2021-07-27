import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageColor, ImageOps
import os

def face_detection(image, confidence=0.50):

    # load dnn caffe model
    PROTOTXT = "deploy.prototxt.txt"
    MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
    dnn = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    # convert image to array
    converted_image = np.array(image.convert("RGB"))

    # get image dimensions: height and width
    (h, w) = converted_image.shape[:2]

    # resize image to training size
    resized_image = cv2.resize(converted_image, (300, 300))

    # preprocess the image before passing it through DNN
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the preprocessed image
    dnn.setInput(blob)

    # detect regions of faces
    detections = dnn.forward()

    # iterate over all detections
    detected_faces = [] #(startX, startY, endX, endY)
    for i in range(0, detections.shape[2]):
        # extract the confidence of the current detection
        detection_confidence = detections[0, 0, i, 2]
        if detection_confidence > confidence:
            # get face box position (startX, startY, endX, endY)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            face = box.astype("int")
            detected_faces.append(face)
    return detected_faces #(startX, startY, endX, endY)

def blur_faces(image, detected_faces, k_factor, border, border_color):

    # convert image to array
    converted_image = np.array(image.convert("RGB"))

    # get image dimensions: height and width
    (h, w) = converted_image.shape[:2]

    # define kernel dimensions (w and h)
    # ksize.width > 0 & & ksize.width % 2 == 1
    k_factor = k_factor / 4
    ksize_w = int(w * k_factor)
    if ksize_w % 2 != 1:
        ksize_w += 1
    # size.height > 0 & & ksize.height % 2 == 1
    ksize_h = int(h * k_factor)
    if ksize_h % 2 != 1:
        ksize_h += 1

    # blur image
    blured_image = converted_image.copy()

    for face in detected_faces:
        (startX, startY, endX, endY) = face
        face_draw = converted_image[startY:endY, startX:endX]
        blured = cv2.GaussianBlur(face_draw, (ksize_w, ksize_h), 0)
        blured_image[startY:endY, startX:endX] = blured
        # draw rectangle
        if border is True:
            border_thick = int((w * h) / (4*(10**6))) + 1
            color_rgb = ImageColor.getcolor(border_color, "RGB")
            cv2.rectangle(blured_image, (startX, startY), (endX, endY), color_rgb, border_thick)

    return blured_image

def main():

    # define max_width to plot image
    max_width = 500

    # siderbar - parameters
    st.sidebar.header("Parameters:")
    st.sidebar.markdown("#####")
    blur_factor = st.sidebar.slider("Select face blur intensity",0.0, 1.0, 0.8, 0.1)
    st.sidebar.markdown("#####")
    border = st.sidebar.selectbox("Add border", ["No", "Yes"], 0)
    if border == "Yes":
        border_color = st.sidebar.color_picker('Select a color for the face border', '#ff0000')
        border_param = True
    elif border == "No":
        border_color = 0
        border_param = False

    # main page
    st.title("Face Detection App")
    st.text("by Hugo Martins")
    st.markdown("####")
    st.markdown(f"""
        Upload a picture with at least one face to be detected.
        
        Don't forget to configure the parameters to get your desired configuration.""")
    st.markdown("***")

    # uploader button
    st.subheader("Load Your Picture")
    uploaded_image = st.file_uploader("Choose Your Picture", type=['jpg', 'jpeg', 'png'])

    # display uploaded picture
    if uploaded_image is not None:
        image_load_state = st.empty()
        image_load_state.text("Loading picture...")
        image_read = Image.open(uploaded_image)
        image_read = ImageOps.exif_transpose(image_read)
        face_detec = face_detection(image_read)
        image = blur_faces(image_read, face_detec, blur_factor, border_param, border_color)
        image_width = image_read.size[0]
        if image_width > max_width:
            st.image(image, width=max_width)
        else:
            st.image(image, width=image_width)
        if len(face_detec) > 1:
            image_load_state.text(f"Uploaded picture: {len(face_detec)} faces detected")
        else:
            image_load_state.text(f"Uploaded picture: {len(face_detec)} face detected")

    # demo button
    st.markdown("###")
    st.subheader("Or")
    button_demo = st.button("Try a demo")

    # set random demo files
    demo_pictures = []
    directory = "demo_pictures"
    for filename in os.listdir(directory):
        if filename.split(".")[-1] in ['jpg', 'jpeg', 'png']:
            demo_pictures.append(filename)
    rand_pos = np.random.randint(0, len(demo_pictures))
    demo_image = os.path.join(directory, demo_pictures[rand_pos])

    # display demo picture
    if (uploaded_image is None) and (button_demo is True):
        image_load_state = st.text("Loading demo picture...")
        image_read = Image.open(demo_image)
        image_read = ImageOps.exif_transpose(image_read)
        face_detec = face_detection(image_read)
        image = blur_faces(image_read, face_detec, blur_factor, border_param, border_color)
        col1, col2 = st.beta_columns(2)
        desc1 = col1.empty()
        desc2 = col2.empty()
        col1.image(image_read, use_column_width=True)
        col2.image(image, use_column_width=True)
        image_load_state.empty()
        desc1.text(f"Original picture")
        if len(face_detec) > 1:
            desc2.text(f"Demo picture: {len(face_detec)} faces detected")
        else:
            desc2.text(f"Demo picture: {len(face_detec)} face detected")
    elif (uploaded_image is not None) and (button_demo is True):
        st.text("Please, close your uploaded picture to run the demo.")

if __name__ == '__main__':
    st.set_page_config(page_title = "Face Detection by Hugo Martins")
    main()
