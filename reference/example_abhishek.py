import os
import shutil
import pandas as pd
import numpy as np

import ast  # train.csv의 리스트형태 정보만 뽑아올때 유용하게 사용

from sklearn import model_selection
from tqdm import tqdm

DATA_PATH = "/home/kerrykim/jupyter_notebook/8.wheat_detection/data_origin/"
OUTPUT_PATH = "/home/kerrykim/jupyter_notebook/8.wheat_detection/wheat_data/"


# pytorch dataset에서는 __getitem__에서 index를 통해 값을 하나씩 받았다.
# 여기서는 데이터프레임의 row별로 읽으면서 처리하기 위해 iterrows() 함수를 사용한다.
# 실제로는 itertuples()가 더 빠르고 유용한듯
# https://blog.naver.com/ehdry3939/221803824089
def process_data(data, data_type="train"):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row["image_id"]
        bounding_boxes = row["bboxes"]
        yolo_data = []
        for bbox in bounding_boxes:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x + w / 2
            y_center = y + h / 2
            x_center /= 1024.0
            y_center /= 1024.0
            w /= 1024.0
            h /= 1024.0
            # yolo 라벨을 만들때 가장 먼저 앞에 클래스가 들어가는데 여기선 0뿐이다.
            yolo_data.append([0, x_center, y_center, w, h])

        yolo_data = np.array(yolo_data)
        # python f-string, 3.7버전부터 적용됨
        # {}를 쓰면 중간에 내가 원하는 글자를 쓸 수 있구나
        np.savetxt(os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
                   yolo_data,
                   fmt=["%d", "%f", "%f", "%f", "%f"])

        shutil.copyfile(os.path.join(DATA_PATH, f"train/{image_name}.jpg"),
                        os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg"))


if __name__=="__main__":
    df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    df.bbox = df.bbox.apply(ast.literal_eval)
    df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes")

    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    process_data(df_train, data_type="train")
    process_data(df_valid, data_type="validation")

'''
> conda activate ml
> ipython

import ast

l = "[1, 2, 3, 4]"
type(l) #결과는 str

ast.literal_eval(l) #결과는 [1, 2, 3, 4]
type(ast.literl_eval(l)) #결과는 list
'''

'''
**** df = df.groupby("image_id")["bbox"].apply(list) 한 경우 ****

> conda activate global_wheat_detection
> python munge_data.py

image_id
00333207f    [[0, 654, 37, 111], [0, 817, 135, 98], [0, 192...
005b0d8bb    [[765.0, 879.0, 116.0, 79.0], [84.0, 539.0, 15...
006a994f7    [[437.0, 988.0, 98.0, 36.0], [309.0, 527.0, 11...
00764ad5d    [[89.0, 256.0, 113.0, 107.0], [216.0, 282.0, 1...
00b5fefed    [[709.0, 97.0, 204.0, 105.0], [775.0, 250.0, 1...
                                   ...                        
ffb445410    [[0.0, 534.0, 54.0, 118.0], [0.0, 480.0, 38.0,...
ffbf75e5b    [[0, 697, 21, 58], [104, 750, 77, 75], [65, 84...
ffbfe7cc0    [[256.0, 0.0, 64.0, 99.0], [390.0, 0.0, 48.0, ...
ffc870198    [[447.0, 976.0, 78.0, 48.0], [18.0, 141.0, 218...
ffdf83e42    [[306.0, 178.0, 67.0, 88.0], [367.0, 167.0, 63...
Name: bbox, Length: 3373, dtype: object

-> image_id 별로 묶여서 출력이 되고 있다. 즉 3373개의 image_id가 존재한다.
-> bbox에 한번 더 list를 적용해서 []가 하나 더 끼워져있다.


**** df = df.groupby("image_id")["bbox"].apply(list).reset_index(drop=True) ****
> python munge_data.py

0       [[0, 654, 37, 111], [0, 817, 135, 98], [0, 192...
1       [[765.0, 879.0, 116.0, 79.0], [84.0, 539.0, 15...
2       [[437.0, 988.0, 98.0, 36.0], [309.0, 527.0, 11...
3       [[89.0, 256.0, 113.0, 107.0], [216.0, 282.0, 1...
4       [[709.0, 97.0, 204.0, 105.0], [775.0, 250.0, 1...
                              ...                        
3368    [[0.0, 534.0, 54.0, 118.0], [0.0, 480.0, 38.0,...
3369    [[0, 697, 21, 58], [104, 750, 77, 75], [65, 84...
3370    [[256.0, 0.0, 64.0, 99.0], [390.0, 0.0, 48.0, ...
3371    [[447.0, 976.0, 78.0, 48.0], [18.0, 141.0, 218...
3372    [[306.0, 178.0, 67.0, 88.0], [367.0, 167.0, 63...
Name: bbox, Length: 3373, dtype: object

**** df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes") ****

       image_id                                             bboxes
0     00333207f  [[0, 654, 37, 111], [0, 817, 135, 98], [0, 192...
1     005b0d8bb  [[765.0, 879.0, 116.0, 79.0], [84.0, 539.0, 15...
2     006a994f7  [[437.0, 988.0, 98.0, 36.0], [309.0, 527.0, 11...
3     00764ad5d  [[89.0, 256.0, 113.0, 107.0], [216.0, 282.0, 1...
4     00b5fefed  [[709.0, 97.0, 204.0, 105.0], [775.0, 250.0, 1...
...         ...                                                ...
3368  ffb445410  [[0.0, 534.0, 54.0, 118.0], [0.0, 480.0, 38.0,...
3369  ffbf75e5b  [[0, 697, 21, 58], [104, 750, 77, 75], [65, 84...
3370  ffbfe7cc0  [[256.0, 0.0, 64.0, 99.0], [390.0, 0.0, 48.0, ...
3371  ffc870198  [[447.0, 976.0, 78.0, 48.0], [18.0, 141.0, 218...
3372  ffdf83e42  [[306.0, 178.0, 67.0, 88.0], [367.0, 167.0, 63...

[3373 rows x 2 columns]
'''


'''
# 다음은 터미널을 통해 폴더를 하나 만든다.
# > mkdir wheat_data
# > cd wheat_data
# > mkdir images
# > mkdir labels
# > cd images
# > mkdir train
# > mkdir valiation
# > cd ..                   # cd .. 은 상위 디렉토리로 다시 나오기
# > cd labels
# > mkdir train
# > mkdir validation
# > cd ..
# > cd ..
# > tree wheat_data

# > mv wheat_data/images/valiation wheat_data/images/validation #mv를 사용해 폴더이름을 수정
# > tree wheat_data

Yolov5의 라벨은 클래스(0, 1, 2, ..)/x_center/y_center/width/height 순서를 갖어야 한다.

'''


