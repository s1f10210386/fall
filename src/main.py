import cv2
import numpy as np
import math

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

        # 身体の傾き角度を計算
        angle = calculate_body_angle(keypoints, scores)
        if angle is not None:
            print(f"身体の傾き角度: {angle:.2f}度")

            # 転倒の閾値を設定（例えば、45度以上を転倒とみなす）
            FALL_ANGLE_THRESHOLD = 80
            if angle > FALL_ANGLE_THRESHOLD:
                print("転倒を検知しました！")
                # ここでGPT APIの呼び出しや、他のアクションを実行できます

        # 画像レンダリング
        result_image = render(frame, keypoints, scores)

        cv2.namedWindow("image", cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', result_image)
        key = cv2.waitKey(1) & 0xFF
        # Q押下で終了
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

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
    # keypoints = keypoints.astype(int)  # 描画時に整数化する

    return keypoints, scores

def calculate_body_angle(keypoints, scores, threshold=KEYPOINT_THRESHOLD):
    # 左肩、右肩、左腰、右腰のインデックス
    left_shoulder = 5
    right_shoulder = 6
    left_hip = 11
    right_hip = 12

    # キーポイントのスコアが閾値以上であることを確認
    required_keypoints = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(scores[idx] < threshold for idx in required_keypoints):
        return None  # 信頼できるデータがない場合はNoneを返す

    # 左右の肩と腰の平均座標を計算（(y, x) の順序に注意）
    shoulder_center_y = (keypoints[left_shoulder][0] + keypoints[right_shoulder][0]) / 2
    shoulder_center_x = (keypoints[left_shoulder][1] + keypoints[right_shoulder][1]) / 2
    hip_center_y = (keypoints[left_hip][0] + keypoints[right_hip][0]) / 2
    hip_center_x = (keypoints[left_hip][1] + keypoints[right_hip][1]) / 2

    # 身体の中心線のベクトルを計算（肩から腰へのベクトル）
    vector_x = hip_center_x - shoulder_center_x
    vector_y = hip_center_y - shoulder_center_y

    # ベクトルの大きさを計算
    magnitude_vector = math.sqrt(vector_x**2 + vector_y**2)
    if magnitude_vector == 0:
        return None  # ベクトルの大きさが0の場合は角度を計算できない

    # 垂直線との角度を計算（垂直線は (0, 1) とする）
    # ベクトルと垂直線の内積を計算
    dot_product = vector_y  # 垂直線が (0, 1) なので、内積は vector_y

    # コサインを計算（値が -1 から 1 の範囲にあることを確認）
    cos_theta = dot_product / magnitude_vector
    cos_theta = max(min(cos_theta, 1.0), -1.0)  # 誤差対策

    # 角度を計算（度数法）
    angle = math.degrees(math.acos(cos_theta))

    return angle

def render(image, keypoints, scores):
    render_image = image.copy()
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
            cv2.line(render_image, (int(kp_1[1]), int(kp_1[0])), (int(kp_2[1]), int(kp_2[0])), (0,0,255), 2)
    
    for idx, (keypoint, score) in enumerate(zip(keypoints, scores)):
        if score > KEYPOINT_THRESHOLD:
            cv2.circle(render_image, (int(keypoint[1]), int(keypoint[0])), 4, (0,0,255), -1)
    
    return render_image

if __name__ == '__main__':
    main()
