import math
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    from PIL import ImageResampling
    RESAMPLE_FILTER = ImageResampling.LANCZOS
except ImportError:
    RESAMPLE_FILTER = Image.LANCZOS

filename = ''
img1 = None
top = None
image_label = None
result_label = None
v = None  # Radiobutton: moment type (Hu, R, Zernike)
m = None  # Radiobutton: comparison method (1,2,3)

def main():
    global top, image_label, result_label, v, m

    top = tk.Tk()
    top.title('Catch All ImageZ')
    top.geometry('800x500')
    top.minsize(600, 400)

    style = ttk.Style(top)
    style.theme_use("clam")  

    style.configure(
        "BigButton.TButton",
        font=('Arial', 14),
        padding=10,
        foreground="white",     
        background="royalblue"  
    )
    style.map(
        "BigButton.TButton",
        background=[("active", "blue")]  
    )

    style.configure(
        "TRadiobutton",
        font=('Arial', 13),
        padding=5
    )

    top.rowconfigure(0, weight=1)  
    top.rowconfigure(1, weight=0) 
    top.rowconfigure(2, weight=0) 
    top.columnconfigure(0, weight=1)

    image_frame = ttk.Frame(top, borderwidth=2, relief="groove")
    image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    global image_label
    image_label = ttk.Label(image_frame, text="No image selected", anchor="center")
    image_label.pack(expand=True, fill="both")

    bottom_frame = ttk.Frame(top)
    bottom_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

    bottom_frame.columnconfigure(0, weight=1)
    bottom_frame.columnconfigure(1, weight=1)

    button_frame = ttk.Frame(bottom_frame)
    button_frame.grid(row=0, column=0, sticky="w", padx=80, pady=5)

    radio_frame = ttk.Frame(bottom_frame)
    radio_frame.grid(row=0, column=1, sticky="e", padx=10)

    select_btn = ttk.Button(
        button_frame,
        text="Select Image",
        style="BigButton.TButton",
        command=ButtonEvent
    )
    select_btn.pack(side="top", anchor="w", pady=5)

    find_btn = ttk.Button(
        button_frame,
        text="Find Characters",
        style="BigButton.TButton",
        command=ButtonEvent2
    )
    find_btn.pack(side="top", anchor="w", pady=5)

    v = tk.IntVar(value=1)  # default: Hu
    m = tk.IntVar(value=1)  # default: Comparison 1

    ttk.Radiobutton(radio_frame, text='Hu Moment',      variable=v, value=1).pack(anchor="e")
    ttk.Radiobutton(radio_frame, text='R Moment',       variable=v, value=2).pack(anchor="e")
    ttk.Radiobutton(radio_frame, text='Zernike Moment', variable=v, value=3).pack(anchor="e")

    ttk.Radiobutton(radio_frame, text='Comparison Method 1', variable=m, value=1).pack(anchor="e", pady=(10,0))
    ttk.Radiobutton(radio_frame, text='Comparison Method 2', variable=m, value=2).pack(anchor="e")
    ttk.Radiobutton(radio_frame, text='Comparison Method 3', variable=m, value=3).pack(anchor="e")

    global result_label
    result_label = ttk.Label(top, text="", font=('Arial', 12, 'bold'), foreground="blue")
    result_label.grid(row=2, column=0, sticky="n", pady=(5, 10))

    top.mainloop()

def ButtonEvent():
    global img1, filename

    selected = filedialog.askopenfilename(
        initialdir="/Desktop",
        title="Select file",
        filetypes=[("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")]
    )
    if not selected:
        return

    filename = selected
    print("Selected file:", filename)

    image = Image.open(filename)
    width, height = image.size

    max_w, max_h = 400, 250
    if width > max_w:
        ratio = max_w / width
        width = int(width * ratio)
        height = int(height * ratio)
    if height > max_h:
        ratio = max_h / height
        width = int(width * ratio)
        height = int(height * ratio)

    image = image.resize((width, height), RESAMPLE_FILTER)
    img1 = ImageTk.PhotoImage(image)
    image_label.config(image=img1, text="")

def ButtonEvent2():
    global filename

    if not filename:
        messagebox.showinfo("Warning", "Please select image.")
        return

    try:
        img = Image.open(filename).convert("RGBA")
    except (AttributeError, FileNotFoundError):
        messagebox.showinfo("Warning", "Please select image.")
        return

    img_gray = img.convert('L')
    ONE = 255
    a = np.asarray(img_gray)
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)

    label = blob_coloring_8_connected(a_bin, ONE)
    labelDict = add_dict_labelled(label)
    recAtt = find_rectangle(label, labelDict)

    drawing_rect(img, recAtt)
    numberOfLabels = len(labelDict)
    type_moment = ""
    featureVectors = None

    if v.get() == 1:  
        type_moment = "Hu"
        numberOfMoments = 7
        featureVectors = np.empty((numberOfLabels, numberOfMoments), dtype=np.double)
        for i in range(numberOfLabels):
            minx, miny, maxx, maxy = recAtt[0][i], recAtt[1][i], recAtt[2][i], recAtt[3][i]
            resizedIm = resize_rec(im, minx, miny, maxx, maxy)
            featureVectors[i] = calc_moments_hu(resizedIm)

    elif v.get() == 2:  
        type_moment = "R"
        numberOfMoments = 10
        featureVectors = np.empty((numberOfLabels, numberOfMoments), dtype=np.double)
        for i in range(numberOfLabels):
            minx, miny, maxx, maxy = recAtt[0][i], recAtt[1][i], recAtt[2][i], recAtt[3][i]
            resizedIm = resize_rec(im, minx, miny, maxx, maxy)
            featureVectors[i] = calc_moments_r(resizedIm)

    elif v.get() == 3:  
        type_moment = "Zernike"
        numberOfMoments = 12
        featureVectors = np.empty((numberOfLabels, numberOfMoments), dtype=np.double)
        for i in range(numberOfLabels):
            minx, miny, maxx, maxy = recAtt[0][i], recAtt[1][i], recAtt[2][i], recAtt[3][i]
            resizedIm = resize_rec(im, minx, miny, maxx, maxy)
            featureVectors[i] = calc_moments_zernike(resizedIm)

    results = []
    if featureVectors is not None:
        if m.get() == 1:
            results = multiple_comparison(featureVectors, recAtt, type_moment)
        elif m.get() == 2:
            results = multiple_comparison2(featureVectors, recAtt, type_moment)
        elif m.get() == 3:
            results = multiple_comparison3(featureVectors, recAtt, type_moment)

        write_assumptions(recAtt, img, results)

    recognized_text = "Characters found: " + " ".join(map(str, results))
    result_label.config(text=recognized_text)

def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros_like(im, dtype=np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out

def blob_coloring_8_connected(bim, ONE):
    max_label = 10000
    nrow, ncol = bim.shape
    im = np.full((nrow, ncol), max_label, dtype=int)
    a = np.arange(max_label, dtype=int)

    color_map = np.zeros((max_label, 3), dtype=np.uint8)
    color_im = np.zeros((nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, dtype=np.uint8)

    k = 0
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            c = bim[i][j]
            label_l = im[i - 1][j]
            label_u = im[i][j - 1]
            label_d = im[i - 1][j - 1]
            label_r = im[i + 1][j - 1]
            if c == ONE:
                min_label = min(label_u, label_l, label_d, label_r)
                if min_label == max_label:
                    k += 1
                    im[i][j] = k
                else:
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label:
                        update_array(a, min_label, label_u)
                    if min_label != label_l and label_l != max_label:
                        update_array(a, min_label, label_l)
                    if min_label != label_d and label_d != max_label:
                        update_array(a, min_label, label_d)
                    if min_label != label_r and label_r != max_label:
                        update_array(a, min_label, label_r)

    # union-find pass
    for i in range(k + 1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    for i in range(nrow):
        for j in range(ncol):
            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    color_im[i][j] = (0, 0, 0)
                else:
                    color_im[i][j] = color_map[im[i][j]]

    return color_im

def update_array(a, label1, label2):
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:
            break

def add_dict_labelled(color_im):
    nrow, ncol, _ = color_im.shape
    labelDict = {}
    for i in range(nrow):
        for j in range(ncol):
            if not np.all(color_im[i][j] == 0):
                key = rgb_to_hash(color_im[i][j])
                if key not in labelDict:
                    labelDict[key] = [i, j]
    labelDict = add_dict_labelled2(labelDict)
    return labelDict

def add_dict_labelled2(label_dict):
    label_dict2 = {}
    number_of_labels = 0
    for key in label_dict:
        label_dict2[key] = number_of_labels
        number_of_labels += 1
    return label_dict2

def find_rectangle(color_im, label_dict):
    number_of_labels = len(label_dict)
    rec_att = np.zeros((4, number_of_labels), dtype=float)

    nrow, ncol, _ = color_im.shape
    rec_att[0, :] = 10000
    rec_att[1, :] = 10000

    for i in range(nrow):
        for j in range(ncol):
            if not np.all(color_im[i][j] == 0):
                current = rgb_to_hash(color_im[i][j])
                currentIndex = label_dict[current]
                if i < rec_att[0][currentIndex]:
                    rec_att[0][currentIndex] = i
                if j < rec_att[1][currentIndex]:
                    rec_att[1][currentIndex] = j
                if i > rec_att[2][currentIndex]:
                    rec_att[2][currentIndex] = i
                if j > rec_att[3][currentIndex]:
                    rec_att[3][currentIndex] = j
    return rec_att

def rgb_to_hash(rgb_array):
    return int(rgb_array[0]*1000000 + rgb_array[1]*1000 + rgb_array[2])

def drawing_rect(img, rec_att):
    draw = ImageDraw.Draw(img)
    ncol = rec_att.shape[1]
    for j in range(ncol):
        left   = rec_att[1][j]
        top    = rec_att[0][j]
        right  = rec_att[3][j]
        bottom = rec_att[2][j]
        shape  = [(left, top), (right, bottom)]
        draw.rectangle(shape, outline="red", width=1)

def resize_rec(im, min_x, min_y, max_x, max_y):
    box = (min_y, min_x, max_y, max_x)
    im2 = im.crop(box)
    im3 = im2.resize((21, 21), RESAMPLE_FILTER)
    return im3

def write_assumptions(rec_att, img, results):
    ncol = rec_att.shape[1]
    for j in range(ncol):
        x1 = rec_att[1][j]
        y1 = rec_att[0][j]
        x2 = rec_att[3][j]
        y2 = rec_att[2][j]

        txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
        try:
            fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28, encoding="unic")
        except OSError:
            fnt = ImageFont.load_default()

        d = ImageDraw.Draw(txt)
        text_x = (x1 + x2 - 15) / 2
        text_y = y1 - 22
        d.text((text_x, text_y), str(results[j]), font=fnt, fill=(0, 0, 0, 255))

        img = Image.alpha_composite(img, txt)


def calc_moments_hu(resized_image):
    f = np.asarray(resized_image)
    nrow, ncol = f.shape[:2]

    raw_moments = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            for x in range(nrow):
                for y in range(ncol):
                    raw_moments[i][j] += (x**i)*(y**j)*f[x][y]

    central_moments = [[0]*4 for _ in range(4)]
    xZero = raw_moments[1][0]/(raw_moments[0][0] if raw_moments[0][0] != 0 else 1)
    yZero = raw_moments[0][1]/(raw_moments[0][0] if raw_moments[0][0] != 0 else 1)

    for i in range(4):
        for j in range(4):
            for x in range(nrow):
                for y in range(ncol):
                    central_moments[i][j] += ((x - xZero)**i)*((y - yZero)**j)*f[x][y]

    normalized_central = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if central_moments[0][0] != 0:
                normalized_central[i][j] = central_moments[i][j]/(central_moments[0][0]**(1+((i+j)/2)))
            else:
                normalized_central[i][j] = 0

    H1 = normalized_central[2][0] + normalized_central[0][2]
    H2 = (normalized_central[2][0] - normalized_central[0][2])**2 + 4*(normalized_central[1][1]**2)
    H3 = ((normalized_central[3][0] - 3*normalized_central[1][2])**2 +
          (3*normalized_central[2][1] - normalized_central[0][3])**2)
    H4 = ((normalized_central[3][0] + normalized_central[1][2])**2 +
          (normalized_central[2][1] + normalized_central[0][3])**2)
    H5 = ((normalized_central[3][0] - 3*normalized_central[1][2]) *
          (normalized_central[3][0] + normalized_central[1][2]) *
          ((normalized_central[3][0] + normalized_central[1][2])**2 -
           3*(normalized_central[2][1] + normalized_central[0][3])**2) +
          (3*normalized_central[2][1] - normalized_central[0][3]) *
          (normalized_central[2][1] + normalized_central[0][3]) *
          (3*(normalized_central[3][0] + normalized_central[1][2])**2 -
           (normalized_central[2][1] + normalized_central[0][3])**2))
    H6 = ((normalized_central[2][0] - normalized_central[0][2]) *
          ((normalized_central[3][0] + normalized_central[1][2])**2 -
           (normalized_central[2][1] + normalized_central[0][3])**2) +
          4*normalized_central[1][1]*(normalized_central[3][0] + normalized_central[1][2])*
          (normalized_central[2][1] + normalized_central[0][3]))
    H7 = -(((3*normalized_central[2][1] - normalized_central[0][3])*
             (normalized_central[3][0] + normalized_central[1][2]) *
             ((normalized_central[3][0] + normalized_central[1][2])**2 -
              3*(normalized_central[2][1] + normalized_central[0][3])**2)) -
            ((normalized_central[3][0] - 3*normalized_central[1][2])*
             (normalized_central[2][1] + normalized_central[0][3])*
             (3*(normalized_central[3][0] + normalized_central[1][2])**2 -
              (normalized_central[2][1] + normalized_central[0][3])**2)))

    return [H1,H2,H3,H4,H5,H6,H7]

def calc_moments_r(resized_image):
    hu_m = calc_moments_hu(resized_image)
    h1, h2, h3, h4, h5, h6, h7 = hu_m
    if h1 == 0:
        h1 = 1e-9
    if h3 == 0:
        h3 = 1e-9
    if h4 == 0:
        h4 = 1e-9
    if h2 == 0:
        h2 = 1e-9
    if h5 == 0:
        h5 = 1e-9

    r1 = math.sqrt(h2)/h1
    r2 = (h1 + math.sqrt(h2)) / (h1 - math.sqrt(h2)) if abs(h1 - math.sqrt(h2)) > 1e-12 else 0
    r3 = math.sqrt(h3)/math.sqrt(h4)
    r4 = math.sqrt(h3)/math.sqrt(abs(h5))
    r5 = math.sqrt(h4)/math.sqrt(abs(h5))
    r6 = abs(h6)/(h1*h3)
    r7 = abs(h6)/(h1*math.sqrt(abs(h5)))
    r8 = abs(h6)/(h3*math.sqrt(abs(h2)))
    r9 = abs(h6)/(math.sqrt(h2*abs(h5)))
    r10 = abs(h5)/(h3*h4)

    return [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10]

def calc_moments_zernike(resized_image):
    z11 = math.sqrt(zernike_ZRnm(resized_image,1,1)**2 + zernike_ZInm(resized_image,1,1)**2)
    z22 = math.sqrt(zernike_ZRnm(resized_image,2,2)**2 + zernike_ZInm(resized_image,2,2)**2)
    z31 = math.sqrt(zernike_ZRnm(resized_image,3,1)**2 + zernike_ZInm(resized_image,3,1)**2)
    z33 = math.sqrt(zernike_ZRnm(resized_image,3,3)**2 + zernike_ZInm(resized_image,3,3)**2)
    z42 = math.sqrt(zernike_ZRnm(resized_image,4,2)**2 + zernike_ZInm(resized_image,4,2)**2)
    z44 = math.sqrt(zernike_ZRnm(resized_image,4,4)**2 + zernike_ZInm(resized_image,4,4)**2)
    z51 = math.sqrt(zernike_ZRnm(resized_image,5,1)**2 + zernike_ZInm(resized_image,5,1)**2)
    z53 = math.sqrt(zernike_ZRnm(resized_image,5,3)**2 + zernike_ZInm(resized_image,5,3)**2)
    z55 = math.sqrt(zernike_ZRnm(resized_image,5,5)**2 + zernike_ZInm(resized_image,5,5)**2)
    z62 = math.sqrt(zernike_ZRnm(resized_image,6,3)**2 + zernike_ZInm(resized_image,6,2)**2)
    z64 = math.sqrt(zernike_ZRnm(resized_image,6,4)**2 + zernike_ZInm(resized_image,6,4)**2)
    z66 = math.sqrt(zernike_ZRnm(resized_image,6,6)**2 + zernike_ZInm(resized_image,6,6)**2)

    return [z11,z22,z31,z33,z42,z44,z51,z53,z55,z62,z64,z66]

def zernike_rnm(n,m,pij):
    rnm=0
    for i in range(int((n-abs(m))/2)):
        rnm += ((-1)**i * pij**(n-2*i) * math.factorial(n-i) /
                (math.factorial(i)*math.factorial(int((n+abs(m))/2)-i)*math.factorial(int((n-abs(m))/2)-i)))
    return rnm

def zernike_ZRnm(resized_image,n,m):
    f = np.asarray(resized_image)
    nrow, ncol = f.shape[:2]
    zr = 0.0
    for i in range(nrow):
        for j in range(ncol):
            xi = (math.sqrt(2)/(nrow-1))*i - 1/math.sqrt(2)
            yj = (math.sqrt(2)/(ncol-1))*j - 1/math.sqrt(2)
            pij = math.sqrt(xi**2 + yj**2)
            qij = math.atan2(yj, xi)
            zr += (f[i][j]*zernike_rnm(n,m,pij)*math.cos(m*qij)*(2/(nrow*math.sqrt(2)))**2)
    zr *= (n+1)/math.pi
    return zr

def zernike_ZInm(resized_image,n,m):
    f = np.asarray(resized_image)
    nrow, ncol = f.shape[:2]
    zi = 0.0
    for i in range(nrow):
        for j in range(ncol):
            xi = (math.sqrt(2)/(nrow-1))*i - 1/math.sqrt(2)
            yj = (math.sqrt(2)/(ncol-1))*j - 1/math.sqrt(2)
            pij = math.sqrt(xi**2 + yj**2)
            qij = math.atan2(yj, xi)
            zi += (f[i][j]*zernike_rnm(n,m,pij)*math.sin(m*qij)*(2/(nrow*math.sqrt(2)))**2)
    zi *= -((n+1)/math.pi)
    return zi


def multiple_comparison(feature_vectors, recAtt, type_moment):
    numberOfSource = 10
    number_of_label = feature_vectors.shape[0]
    number_of_moment = feature_vectors.shape[1]
    alignment = ['0','1','2','3','4','5','6','7','8','9']

    for k in range(number_of_label):
        for l in range(number_of_moment):
            if feature_vectors[k][l] != 0:
                feature_vectors[k][l] = -np.sign(feature_vectors[k][l]) * np.log10(np.abs(feature_vectors[k][l]))

    most_relevant = np.empty(shape=(number_of_label), dtype=int)
    for i in range(number_of_label):
        distances = np.empty(shape=(number_of_label, numberOfSource), dtype=np.double)
        for j in range(numberOfSource):
            fname = "Database/source" + type_moment + str(j) + ".npy"
            momentSource = np.load(fname)
            characterOfSource = momentSource.shape[0]
            numberOfMoments = momentSource.shape[1]

            for kk in range(characterOfSource):
                for ll in range(numberOfMoments):
                    if momentSource[kk][ll] != 0:
                        momentSource[kk][ll] = -np.sign(momentSource[kk][ll]) * np.log10(np.abs(momentSource[kk][ll]))

            distances0 = np.zeros(shape=(characterOfSource), dtype=np.double)
            for k_ in range(characterOfSource):
                dis = 0
                for idx in range(numberOfMoments):
                    dis += (momentSource[k_][idx] - feature_vectors[i][idx])**2
                totalDis = math.sqrt(dis)
                distances0[k_] = totalDis

            distances[i][j] = sum(distances0)/len(distances0)
        most_relevant[i] = alignment[np.argmin(distances[i])]
    return most_relevant

def multiple_comparison2(feature_vectors, rec_att, type_moment):
    number_of_source = 10
    number_of_label = feature_vectors.shape[0]
    number_of_moment = feature_vectors.shape[1]
    alignment = ['0','1','2','3','4','5','6','7','8','9']

    for k in range(number_of_label):
        for l in range(number_of_moment):
            if feature_vectors[k][l] != 0:
                feature_vectors[k][l] = -np.sign(feature_vectors[k][l]) * np.log10(np.abs(feature_vectors[k][l]))

    most_relevant = np.empty(shape=(number_of_label), dtype=int)
    for i in range(number_of_label):
        all_distance_data = np.full((number_of_source, number_of_source, 30), 1000, dtype=np.double)

        for j in range(number_of_source):
            fname = "Database/source" + type_moment + str(j) + ".npy"
            momentSource = np.load(fname)
            characterOfSource = momentSource.shape[0]
            numberOfMoments = momentSource.shape[1]

            for c in range(characterOfSource):
                for ll in range(numberOfMoments):
                    if momentSource[c][ll] != 0:
                        momentSource[c][ll] = -np.sign(momentSource[c][ll]) * np.log10(np.abs(momentSource[c][ll]))

            for k_ in range(characterOfSource):
                dis = 0
                for idx in range(numberOfMoments):
                    dis += (momentSource[k_][idx] - feature_vectors[i][idx])**2
                dis = math.sqrt(dis)

                if k_ < 30:
                    all_distance_data[i][j][k_] = dis

        knn_array = np.full(shape=9, fill_value=1000.0, dtype=np.double)
        knn_array_info = np.full(shape=9, fill_value=-1, dtype=int)
        sizeOfKnn = knn_array.shape[0]

        for y in range(number_of_source):
            for z in range(30):
                val = all_distance_data[i][y][z]
                if val != 1000:
                    for t in range(sizeOfKnn):
                        if val < knn_array[t]:
                            tempValue = knn_array[t]
                            tempInfo = knn_array_info[t]
                            knn_array[t] = val
                            knn_array_info[t] = y

                            if t != sizeOfKnn - 1:
                                for u in range(sizeOfKnn - t - 2):
                                    knn_array[sizeOfKnn - u - 1] = knn_array[sizeOfKnn - u - 2]
                                    knn_array_info[sizeOfKnn - u - 1] = knn_array_info[sizeOfKnn - u - 2]
                                knn_array[t + 1] = tempValue
                                knn_array_info[t + 1] = tempInfo
                            break

        howManyHit = np.zeros(10, dtype=int)
        for p in range(sizeOfKnn):
            if 0 <= knn_array_info[p] < 10:
                howManyHit[knn_array_info[p]] += 1

        maxNumber = howManyHit[0]
        maxNumberIndex = 0
        for p in range(9):
            if howManyHit[p+1] > maxNumber:
                maxNumber = howManyHit[p+1]
                maxNumberIndex = p+1

        most_relevant[i] = maxNumberIndex
    return most_relevant

def multiple_comparison3(feature_vectors, rec_att, type_moment):
    number_of_source = 10
    number_of_label = feature_vectors.shape[0]
    number_of_moment = feature_vectors.shape[1]
    alignment = ['0','1','2','3','4','5','6','7','8','9']

    for k in range(number_of_label):
        for l in range(number_of_moment):
            if feature_vectors[k][l] != 0:
                feature_vectors[k][l] = -np.sign(feature_vectors[k][l]) * np.log10(np.abs(feature_vectors[k][l]))

    most_relevant = np.empty(shape=(number_of_label), dtype=int)
    for i in range(number_of_label):
        distances = np.empty(shape=(number_of_label, number_of_source), dtype=np.double)
        for j in range(number_of_source):
            fname = "Database/source" + type_moment + str(j) + ".npy"
            momentSource = np.load(fname)
            characterOfSource = momentSource.shape[0]
            numberOfMoments = momentSource.shape[1]

            for c in range(characterOfSource):
                for ll in range(numberOfMoments):
                    if momentSource[c][ll] != 0:
                        momentSource[c][ll] = -np.sign(momentSource[c][ll]) * np.log10(np.abs(momentSource[c][ll]))

            distance1 = sys.maxsize
            for k_ in range(characterOfSource):
                ratio_sum = 0.0
                for idx in range(numberOfMoments):
                    denom = momentSource[k_][idx]
                    if denom != 0:
                        ratio_val = abs((momentSource[k_][idx] - feature_vectors[i][idx]) / denom) * 100
                    else:
                        ratio_val = 0
                    ratio_sum += ratio_val

                if ratio_sum < distance1:
                    distance1 = ratio_sum

            distances[i][j] = distance1
        most_relevant[i] = alignment[np.argmin(distances[i])]
    return most_relevant


if __name__ == '__main__':
    main()