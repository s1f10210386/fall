import cv2
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

KEYPOINT_THRESHOLD = 0.2

def main():
    # TensorFlow Hubから単一人物用のMoveNetモデルをダウンロード
    model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
    movenet = model.signatures['serving_default']
    
    # カメラ設定
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        # 推論実行
        keypoints, scores = run_inference(movenet, frame)

        # 画像レンダリング
        result_image = render(frame, keypoints, scores)

        cv2.namedWindow("image", cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', result_image)
        key = cv2.waitKey(1) & 0xFF
        # Q押下で終了
        if key == ord('q'):
            break
    
    cap.release()

def run_inference(model, image):
    # 画像の前処理
    input_image = cv2.resize(image, dsize=(192, 192))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # 推論実行・結果取得
    outputs = model(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    
    # 結果の解析
    keypoints_with_scores = np.squeeze(keypoints_with_scores)
    
    keypoints = keypoints_with_scores[:, :2]
    scores = keypoints_with_scores[:, 2]
    
    # キーポイントを画像座標に変換
    image_height, image_width = image.shape[:2]
    keypoints[:, 0] *= image_height  # y座標
    keypoints[:, 1] *= image_width   # x座標
    keypoints = keypoints.astype(int)
    
    return keypoints, scores

def render(image, keypoints, scores):
    render = image.copy()
    # 接続するキーポイントの組
    kp_links = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),
        (8,10),(11,12),(5,11),(11,13),(13,15),(6,12),(12,14),(14,16)
    ]
    for kp_idx_1, kp_idx_2 in kp_links:
        kp_1 = keypoints[kp_idx_1]
        kp_2 = keypoints[kp_idx_2]
        score_1 = scores[kp_idx_1]
        score_2 = scores[kp_idx_2]
        if score_1 > KEYPOINT_THRESHOLD and score_2 > KEYPOINT_THRESHOLD:
            cv2.line(render, (kp_1[1], kp_1[0]), (kp_2[1], kp_2[0]), (0,0,255), 2)
    
    for idx, (keypoint, score) in enumerate(zip(keypoints, scores)):
        if score > KEYPOINT_THRESHOLD:
            cv2.circle(render, (keypoint[1], keypoint[0]), 4, (0,0,255), -1)
    
    return render

if __name__ == '__main__':
    main()
