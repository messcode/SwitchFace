from FaceDetect import *
fname = "image\\iro3.jpg"  # image path 
lenna =  "image\\Lenna.png"
out = "output\\"

# Mark on lenna
lenna = cv2.imread(lenna, cv2.IMREAD_COLOR)
marked_lenna = annotate_landmarks(lenna, get_landmarks(lenna)[0])
cv2.imwrite(out + "marked_lenna.png", marked_lenna)



img = cv2.imread(fname, cv2.IMREAD_COLOR) 
landmarks = get_landmarks(img)  
mask = get_face_mask(img, landmarks[0])
mask1 = get_face_mask(img, landmarks[1])
M = transformation_from_points(landmarks[1][ALIGN_POINTS], 
                               landmarks[0][ALIGN_POINTS])
affined_mask = warp_im(mask, M, img.shape)
marked_img = annotate_landmarks(img, get_landmarks(img)[0], font_scale = 0)
marked_img = annotate_landmarks(marked_img, get_landmarks(img)[1], font_scale = 0)

combined_mask = affined_mask + mask
warped_img = warp_im(img, M, img.shape)
warped_corrected_img = correct_colours(img, warped_img, landmarks[1])
switched_img = switch_face(fname)
# imwrite always expect [0, 255] unsigned integer while imshow can handle[0, 1] float 
# and [0, 255]. 
cv2.imwrite(out + "mask.jpg", to_uint8(mask * 255))
cv2.imwrite(out + "affined_mask.jpg", to_uint8(affined_mask * 255)) 
cv2.imwrite(out + "combined_mask.jpg", to_uint8(combined_mask * 255)) 
cv2.imwrite(out + "twomasks.jpg", to_uint8((mask + mask1) * 255))
cv2.imwrite(out + "warped_img.jpg", warped_img)
cv2.imwrite(out + "switched_face.jpg", switched_img)  
cv2.imwrite(out + "marked_img.jpg", marked_img)                                
cv2.imwrite(out + "warped_corrected_img.jpg", warped_corrected_img)