from math import tan, cos, radians
import cv2

def p7(image_in, hough_image_in, hough_thresh): #return line_image_out
    num_rows_h = len(hough_image_in)
    num_cols_h = len(hough_image_in[0])

    lines = []
    for rho in range(num_rows_h):
        for theta in range(num_cols_h):
            if hough_image_in[rho][theta] > hough_thresh:
                lines.append((rho,theta,hough_image_in[rho][theta]))

    line_image_out = image_in.copy()
    rho_max = (num_rows_h - 1) / 2
    for (rho,theta,val) in lines:
        rho -= rho_max
        theta = radians(-theta)
        pt1 = (2**32, 2**32)
        pt2 = (2**32, 2**32)
        pt3 = (2**32, 2**32)
        pt4 = (2**32, 2**32)
        if (tan(theta) * cos(theta)) != 0:
            pt1 = (int(-rho/(tan(theta)*cos(theta))), 0)
        if cos(theta) != 0:
            pt2 = (0, int(rho/cos(theta)))
        if cos(theta) != 0 and tan(theta) != 0:
            pt3 = (int((num_rows_h - rho/cos(theta))/(tan(theta))), num_rows_h)
        if cos(theta) != 0:
            pt4 = (num_cols_h, int(tan(theta)*num_cols_h + rho/cos(theta)))
        try:
            cv2.line(line_image_out, pt1, pt2, (255,255,255), 2)
        except:
            try:
                cv2.line(line_image_out, pt1, pt3, (255,255,255), 2)
            except:
                try:
                    cv2.line(line_image_out, pt1, pt4, (255,255,255), 2)
                except:
                    try:
                        cv2.line(line_image_out, pt2, pt3, (255,255,255), 2)
                    except:
                        try:
                            cv2.line(line_image_out, pt2, pt4, (255,255,255), 2)
                        except:
                            try:
                                cv2.line(line_image_out, pt3, pt4, (255,255,255), 2)
                            except:
                                pass
    return line_image_out
