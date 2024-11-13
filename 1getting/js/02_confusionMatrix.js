// 02_confusionMatrix.js

// 혼동행렬 그리기
async function run() {
  // multi-class 분류

  const labels = tf.tensor1d([0,1,2])       // 정답
  const predictions = tf.tensor1d([1,1,2])  // 모델의 예측값

  result = await tfvis.metrics.confusionMatrix(labels, predictions)
  console.log(result)

  // [[0,1,0], [0,1,0], [0,0,1]] : 행은 실제 클래스, 열은 예측된 클래스
  // 첫번째행 [0,1,0] (실제 클래스 0) : 0이 0으로 예측된 횟수 0, 0이 1로 예측된 횟수 1, 0이 2로 예측된 횟수 0
  // 두번째행 : 1이 0으로 예측된 횟수 0, 1이 1로 예측된 횟수 1, 1이 2로 예측된 횟수 0
  // 세번째행 : 2가 0으로 예측된 횟수 0, 2가 1로 예측된 횟수 0, 2가 2로 예측된 횟수 1

  // 시각화
  const surface = {
    name : 'Confusion Matrix', 
    tab : 'Charts'
  }

  // 공식문서에서 데이터를 values에 담으라고 되어 있음
  const data = {
    values : result
  }

  // 시각화 
  tfvis.render.confusionMatrix(surface, data)
}

run();