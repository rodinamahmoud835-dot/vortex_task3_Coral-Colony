{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red195\green123\blue90;\red19\green20\blue21;\red174\green176\blue183;
\red89\green158\blue96;\red71\green149\blue242;\red117\green114\blue185;\red152\green54\blue29;\red38\green157\blue169;
}
{\*\expandedcolortbl;;\csgenericrgb\c76471\c48235\c35294;\csgenericrgb\c7451\c7843\c8235;\csgenericrgb\c68235\c69020\c71765;
\csgenericrgb\c34902\c61961\c37647;\csgenericrgb\c27843\c58431\c94902;\csgenericrgb\c45882\c44706\c72549;\csgenericrgb\c59608\c21176\c11373;\csgenericrgb\c14902\c61569\c66275;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\fs26 \cf2 \cb3 import \cf4 cv2\
\cf2 import \cf4 os\
\cf2 import \cf4 numpy \cf2 as \cf4 np\
\
OneYearImage = cv2.imread(\cf5 "Resources/OneYearImage.jpg"\cf4 )\
\
FULL_MASK  = cv2.imread(\cf5 "Resources/FULL_MASK.png"\cf4 ,  cv2.IMREAD_GRAYSCALE)\
WHITE_MASK = cv2.imread(\cf5 "Resources/WHITE_MASK.png"\cf4 , cv2.IMREAD_GRAYSCALE)\
\
PinkMask = cv2.imread(\cf5 "Resources/PINK_MASK.png"\cf4 , cv2.IMREAD_GRAYSCALE)\
\cf2 if \cf4 PinkMask \cf2 is None\cf4 :\
    PinkMask = cv2.imread(\cf5 "Resources/Pink_Mask.png"\cf4 , cv2.IMREAD_GRAYSCALE)\
\
corals = [\
    (\cf5 "coral1"\cf4 , cv2.imread(\cf5 "Resources/coral1.jpg"\cf4 )),\
    (\cf5 "coral2"\cf4 , cv2.imread(\cf5 "Resources/coral2.jpg"\cf4 )),\
    (\cf5 "coral3"\cf4 , cv2.imread(\cf5 "Resources/coral3.jpg"\cf4 )),\
    (\cf5 "coral4"\cf4 , cv2.imread(\cf5 "Resources/coral4.jpg"\cf4 )),\
    (\cf5 "coral5"\cf4 , cv2.imread(\cf5 "Resources/coral5.jpg"\cf4 )),\
    (\cf5 "coral6"\cf4 , cv2.imread(\cf5 "Resources/coral6.jpeg"\cf4 )),\
]\
\
\cf2 def \cf6 must_load\cf4 (img, name):\
    \cf2 if \cf4 img \cf2 is None\cf4 :\
        \cf2 raise \cf7 FileNotFoundError\cf4 (\cf5 f"Missing: \cf2 \{\cf4 name\cf2 \}\cf5 "\cf4 )\
\
must_load(OneYearImage, \cf5 "OneYearImage.jpg"\cf4 )\
must_load(FULL_MASK, \cf5 "FULL_MASK.png"\cf4 )\
must_load(WHITE_MASK, \cf5 "WHITE_MASK.png"\cf4 )\
must_load(PinkMask, \cf5 "PINK_MASK / Pink_Mask.png"\cf4 )\
\cf2 for \cf4 n, im \cf2 in \cf4 corals:\
    must_load(im, n)\
\
OUTPUT_DIR = \cf5 "Resources/output_task3"\
\cf4 os.makedirs(OUTPUT_DIR, \cf8 exist_ok\cf4 =\cf2 True\cf4 )\
\
USE_ALIGNMENT = \cf2 True\
\
\cf4 PINK_LO  = np.array([\cf9 110\cf4 , \cf9 10\cf4 , \cf9 15\cf4 ], \cf8 dtype\cf4 =np.uint8)\
PINK_HI  = np.array([\cf9 179\cf4 , \cf9 255\cf4 , \cf9 255\cf4 ], \cf8 dtype\cf4 =np.uint8)\
\
WHITE_LO = np.array([\cf9 0\cf4 , \cf9 0\cf4 , \cf9 170\cf4 ], \cf8 dtype\cf4 =np.uint8)\
WHITE_HI = np.array([\cf9 179\cf4 , \cf9 55\cf4 , \cf9 255\cf4 ], \cf8 dtype\cf4 =np.uint8)\
\
MIN_DAMAGE   = \cf9 450\
\cf4 MIN_BLEACH   = \cf9 350\
\cf4 MIN_RECOVERY = \cf9 180\
\
\cf2 def \cf6 binarize\cf4 (m):\
    \cf2 return \cf4 ((m > \cf9 127\cf4 ).astype(np.uint8)) * \cf9 255\
\
\cf2 def \cf6 clean\cf4 (mask, k=\cf9 7\cf4 ):\
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))\
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, \cf8 iterations\cf4 =\cf9 1\cf4 )\
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, \cf8 iterations\cf4 =\cf9 2\cf4 )\
    \cf2 return \cf4 mask\
\
\cf2 def \cf6 align_ecc\cf4 (new_img, old_img, focus_mask):\
    old_g = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / \cf9 255.0\
    \cf4 new_g = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / \cf9 255.0\
\
    \cf4 warp = np.eye(\cf9 2\cf4 , \cf9 3\cf4 , \cf8 dtype\cf4 =np.float32)\
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, \cf9 200\cf4 , \cf9 1e-6\cf4 )\
\
    m = (focus_mask > \cf9 0\cf4 ).astype(np.uint8)\
    \cf2 try\cf4 :\
        cv2.findTransformECC(old_g, new_g, warp, cv2.MOTION_AFFINE, criteria, \cf8 inputMask\cf4 =m, \cf8 gaussFiltSize\cf4 =\cf9 5\cf4 )\
        \cf2 return \cf4 cv2.warpAffine(\
            new_img, warp, (old_img.shape[\cf9 1\cf4 ], old_img.shape[\cf9 0\cf4 ]),\
            \cf8 flags\cf4 =cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,\
            \cf8 borderMode\cf4 =cv2.BORDER_REPLICATE\
        )\
    \cf2 except \cf4 cv2.error:\
        \cf2 return \cf4 new_img\
\
\cf2 def \cf6 draw_boxes\cf4 (img, mask, color, label, min_area):\
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\
    count = \cf9 0\
    \cf2 for \cf4 c \cf2 in \cf4 cnts:\
        area = cv2.contourArea(c)\
        \cf2 if \cf4 area < min_area:\
            \cf2 continue\
        \cf4 x, y, w, h = cv2.boundingRect(c)\
        cv2.rectangle(img, (x, y), (x+w, y+h), color, \cf9 3\cf4 )\
        cv2.putText(img, label, (x, \cf7 max\cf4 (\cf9 0\cf4 , y-\cf9 8\cf4 )),\
                    cv2.FONT_HERSHEY_SIMPLEX, \cf9 0.6\cf4 , color, \cf9 2\cf4 )\
        count += \cf9 1\
    \cf2 return \cf4 count\
\
old_full  = clean(binarize(FULL_MASK), \cf9 5\cf4 )\
old_pink  = clean(binarize(PinkMask), \cf9 5\cf4 )\
old_white = clean(binarize(WHITE_MASK), \cf9 5\cf4 )\
\
focus = cv2.dilate(old_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (\cf9 13\cf4 , \cf9 13\cf4 )), \cf8 iterations\cf4 =\cf9 1\cf4 )\
\
H, W = OneYearImage.shape[:\cf9 2\cf4 ]\
\
\cf2 for \cf4 name, raw \cf2 in \cf4 corals:\
    new_img = cv2.resize(raw, (W, H))\
\
    \cf2 if \cf4 USE_ALIGNMENT:\
        new_img = align_ecc(new_img, OneYearImage, focus)\
\
    hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)\
\
    new_pink  = cv2.inRange(hsv, PINK_LO,  PINK_HI)\
    new_white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)\
\
    new_pink  = cv2.bitwise_and(new_pink,  old_full)\
    new_white = cv2.bitwise_and(new_white, old_full)\
\
    new_pink  = clean(new_pink,  \cf9 5\cf4 )\
    new_white = clean(new_white, \cf9 5\cf4 )\
\
    new_white = cv2.bitwise_and(new_white, cv2.bitwise_not(new_pink))\
\
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)\
    gray = cv2.GaussianBlur(gray, (\cf9 5\cf4 , \cf9 5\cf4 ), \cf9 0\cf4 )\
    roi = cv2.bitwise_and(gray, gray, \cf8 mask\cf4 =old_full)\
    _, new_shape = cv2.threshold(roi, \cf9 0\cf4 , \cf9 255\cf4 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)\
    new_shape = cv2.bitwise_and(new_shape, old_full)\
    new_shape = clean(new_shape, \cf9 7\cf4 )\
\
    damage = cv2.bitwise_and(old_full, cv2.bitwise_not(new_shape))\
    damage = clean(damage, \cf9 9\cf4 )\
\
    bleaching = cv2.bitwise_and(old_pink, new_white)\
    bleaching = cv2.bitwise_and(bleaching, new_shape)\
    bleaching = clean(bleaching, \cf9 9\cf4 )\
\
    recovery = cv2.bitwise_and(old_white, new_pink)\
    recovery = cv2.bitwise_and(recovery, new_shape)\
    recovery = clean(recovery, \cf9 7\cf4 )\
\
    out = new_img.copy()\
    cD = draw_boxes(out, damage,    (\cf9 0\cf4 , \cf9 255\cf4 , \cf9 255\cf4 ), \cf5 "damage"\cf4 ,    MIN_DAMAGE)\
    cB = draw_boxes(out, bleaching, (\cf9 0\cf4 , \cf9 0\cf4 , \cf9 255\cf4 ),   \cf5 "bleaching"\cf4 , MIN_BLEACH)\
    cR = draw_boxes(out, recovery,  (\cf9 255\cf4 , \cf9 0\cf4 , \cf9 0\cf4 ),   \cf5 "recovery"\cf4 ,  MIN_RECOVERY)\
\
    info = \cf5 f"\cf2 \{\cf4 name\cf2 \}\cf5  | D:\cf2 \{\cf4 cD\cf2 \}\cf5  B:\cf2 \{\cf4 cB\cf2 \}\cf5  R:\cf2 \{\cf4 cR\cf2 \}\cf5 "\
    \cf4 cv2.putText(out, info, (\cf9 10\cf4 , \cf9 28\cf4 ), cv2.FONT_HERSHEY_SIMPLEX, \cf9 0.75\cf4 , (\cf9 255\cf4 ,\cf9 255\cf4 ,\cf9 255\cf4 ), \cf9 3\cf4 )\
    cv2.putText(out, info, (\cf9 10\cf4 , \cf9 28\cf4 ), cv2.FONT_HERSHEY_SIMPLEX, \cf9 0.75\cf4 , (\cf9 0\cf4 ,\cf9 0\cf4 ,\cf9 0\cf4 ), \cf9 1\cf4 )\
\
    save_path = os.path.join(OUTPUT_DIR, \cf5 f"\cf2 \{\cf4 name\cf2 \}\cf5 _changes.png"\cf4 )\
    cv2.imwrite(save_path, out)\
    \cf7 print\cf4 (\cf5 "Saved:"\cf4 , save_path)\
\
    cv2.imshow(\cf5 "OLD (1 year ago)"\cf4 , cv2.resize(OneYearImage, (\cf9 900\cf4 , \cf9 550\cf4 )))\
    cv2.imshow(\cf5 "NEW + rectangles"\cf4 ,  cv2.resize(out, (\cf9 900\cf4 , \cf9 550\cf4 )))\
\
    key = cv2.waitKey(\cf9 0\cf4 )\
    \cf2 if \cf4 key == \cf9 27\cf4 :\
        \cf2 break\
\
\cf4 cv2.destroyAllWindows()\
\
}