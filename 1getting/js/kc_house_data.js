// 모델 훈련의 에폭 수 지정
const EPOCHS = 10;
let XnormParams, YnormParams, model;

// TensorFlow.js Visor를 토글하는 함수
function toggleVisor() {
  tfvis.visor().toggle();
}

// 모델을 훈련하는 main 함수
async function train() {
  // CSV 데이터 불러오기
  const HouseSalesDataset = tf.data.csv("kc_house_data.csv", {
    columnConfigs: {    // 열지정
      sqft_living : { isLabel : false },   // 훈련데이터
      price : { isLabel : true }           // 정답데이터
    }, 
    configuredColumnsOnly: true
  })
  console.log(HouseSalesDataset)           // 정보만 찍힌다.

  // 위의 데이터를 매핑시킨다. 훈련 xs, 정답 ys
  // 함수의 인자로 {xs, ys}라는 객체를 받아옴
  // 함수 내에서 xs.sqft_living과 ys.price 값을 추출하여 새로운 객체 {x, y}를 반환
  // map 함수의 결과로, { x: xs.sqft_living, y: ys.price } 형태의 객체들을 담은 새로운 배열이 dataset에 저장
  let dataset = await HouseSalesDataset.map(({xs, ys}) => ({
    x : xs.sqft_living,
    y : ys.price
  }))

  // 데이터를 콘솔에 찍어보자
  console.log(await HouseSalesDataset.take(10).toArray())

  // 모든 요소를 배열로 변환 후 서플
  let dataPoints = await dataset.toArray();

  console.log(dataPoints.length)
  console.log(dataPoints)

  tf.util.shuffle(dataPoints);

  // 데이터를 그래프로 시각화
  // tfvis.render.scatterplot (container, data, opts?)
  let surface = { name: "면적 vs 가격", tab: "Data"};
  let data = { values: [dataPoints] };
  let opts = { xLabel: "면적", yLabel: "가격" };

  tfvis.render.scatterplot (surface, data, opts)

  // 데이터를 텐서로 바꾼다. 
  const featureValues =  dataPoints.map(p => p.x)    // 훈련데이터
  const labelValues = dataPoints.map(p => p.y)       // 정답데이터

  // 특성과 레이블에 대한 텐서 생성
  // 2차원 배열로 tf.tensor2d (values, shape?, dtype?)
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

  featureTensor.print()
  labelTensor.print()

  // 데이터셋을 훈련세트와 테스트세트로 75:25로 분할
  const trainLen = Math.floor(featureTensor.shape[0]*0.75)
  const testLen = Math.floor(featureTensor.shape[0]-trainLen)

  console.log(trainLen, testLen)

  // tf.split()을 사용하여 훈련세트와 테스트세트로 분할
  let [X_train, X_test] = tf.split(featureTensor, [trainLen, testLen])
  let [y_train, y_test] = tf.split(labelTensor, [trainLen, testLen])

  // 데이터 정규화, MinMaxScaling
  // X_train
  const normedXtrainTensor = MinMaxScaling(X_train)

  // 나중에 복원시킬때 사용하기 위해 저장
  XnormParams = { min : normedXtrainTensor.min, max : normedXtrainTensor.max }

  // 우리가 필요한 것은 normedXtrainTensor이다. 더이상 X_train은 필요없다
  X_train.dispose()
  X_train = normedXtrainTensor.tensor;    // 텐서값만 저장

  // y_train 정규화
  const normedyTrainTensor = MinMaxScaling(y_train)
  YnormParams = { min : normedyTrainTensor.min, max : normedyTrainTensor.max }
  y_train.dispose()
  y_train = normedyTrainTensor.tensor;

  // x_test 정규화
  // test값을 정규화할때는 위에서 train에서 계산된 min값과 max값을 그대로 사용한다. 이유는 train했을 때와 test했을 때의 분포가 달라지면 안되기 때문이다. 
  const normedXtestTensor = MinMaxScaling(X_test, XnormParams.min, XnormParams.max)
  X_test.dispose()
  X_test = normedXtestTensor.tensor;

  // y_test 정규화
  const normedyTestTensor = MinMaxScaling(y_test, YnormParams.min, YnormParams.max)
  y_test.dispose()
  y_test = normedyTestTensor.tensor;

  // 정규화된 텐서를 확인해 보자
  X_train.print()
  y_train.print()

  X_test.print()
  y_test.print()

  // 다시 복원시킨다. 
  const denormedTensor = denormalize(y_test, YnormParams.min, YnormParams.max)
  denormedTensor.print()

  // 정규화된 데이터를 시각화 : 시각화를 통하여 실제 데이터와 동일한지 확인해본다
  // 시각화를 위하여 텐서를 배열로 바꾼다. 
  const normedPoints = [];
  const normedFeatureArr = X_train.arraySync();
  const normedLabelArr = y_train.arraySync();
  console.log(normedFeatureArr)
  console.log(normedLabelArr)

  for(let i=0; i<normedFeatureArr.length; i++) {
    normedPoints.push({ x : normedFeatureArr[i], y : normedLabelArr[i] })
  }

  surface = { name: "Normalized - 면적 vs 가격", tab : "Data" }
  data = { values: [normedPoints] }
  opts = { xLabel: "면적", yLabel: "가격" }

  tfvis.render.scatterplot(surface, data, opts);

  // 모델 구성
  // 순차모델: 한 레이어의 출력이 다음 레이어의 입력이 되는 모델
  model = tf.sequential();

  // 은닉층 추가, 1-layer, 입력층은 하나 inputShape: [1] 피처는 하나가지고 있음
  model.add(tf.layers.dense({units:10, inputShape:[1], activation:"relu"}))

  // 출력층 출력은 하나, activation:"linear" 기본값, inputShape생략
  model.add(tf.layers.dense({units:1, activation:"linear"}))

  // compile()에서는 손실함수와 optimizer설정해 준다
  model.compile({loss:'meanSquaredError', optimizer: tf.train.adam(0.01)});
  model.summary();
  tfvis.show.modelSummary({name: "model summary", tab: "model"}, model);

  // 훈련 중 상태 시각화를 위한 콜백함수
  const container = { name: "Training Performance" }
  const metrics = ["loss", "val_loss"]
  const { onEpochEnd, onBatchEnd } = tfvis.show.fitCallbacks(
    container, 
    metrics
  )

  // 모델훈련, history 관습적으로 준다
  // X_train, y_train으로 훈련을 시키는데 {}안에 옵션을 준다
  const history = await model.fit(X_train, y_train, {
    epochs : EPOCHS, 
    batchSize : 32, 
    validationData : [X_test, y_test], 
    callbacks : { onEpochEnd, onBatchEnd }
  })

  // train, validation loss 출력
  console.log(history.history.loss);

  console.log(history.history.loss.pop());
  console.log(history.history.val_loss.pop());

  // train 완료 후
  document.getElementById("text").innerHTML = "학습완료"
  document.getElementById("predict-button").removeAttribute("disabled")

}


function predict() {
  const predictionInput = parseInt(document.getElementById("predict-input").value)
  
  if(isNaN(predictionInput)) {
    alert("숫자를 입력하세요")
  } else {
    tf.tidy(() => {
      // 입력값을 텐서로 바꿈, 1차원 array
      const inputTensor = tf.tensor1d([predictionInput])

      // 정규화
      const normedInput = MinMaxScaling(inputTensor, XnormParams.min, XnormParams.max)

      // 예측한다
      const prediction = model.predict(normedInput.tensor)
      prediction.print()

      // 다시 복원시켜준다
      const denormedPrediction = denormalize(prediction, YnormParams.min, YnormParams.max)
      denormedPrediction.print()

      const output = denormedPrediction.dataSync()[0];
      const output1 = (output*1333).toLocaleString('ko-kr', {maximumFractionDigits:0})
      console.log(output, output1)

      document.getElementById("predict-output").innerHTML = `Predicted Price <br> 달러 : ${output}, 원화 : ${output1}원입니다`;


      // 예측한 결과를 시각화 해보자
      // 가상의 데이터 100개를 만듦
      const [xs, ys] = tf.tidy(() => {
        const normedXs = tf.linspace(0, 1, 100);
        normedXs.print()
        const normedYs = model.predict(normedXs.reshape([100, 1]));
        normedYs.print()
        const denormedXs = denormalize(
          normedXs, 
          XnormParams.min, 
          XnormParams.max
        );
        const denormedYs = denormalize(
          normedYs, 
          YnormParams.min, 
          YnormParams.max
        );
        return [denormedXs.dataSync(), denormedYs.dataSync()];
      })

      const pointsLine = Array.from(xs).map((x, index) => ({
        x : x, 
        y : ys[index]
      }))

      surface = { name: "Predict Line", tab: "Data" }
      data = { values: pointsLine, series: ['Predictions'] }
      opts = { xLabel: 'Square Feet', yLabel: 'Price' }

      tfvis.render.scatterplot(surface, data, opts);

    })
  }
}


// 데이터를 denormalize하는 함수
// 복원 x = 정규화된값*(최대값-최소값)+최소값
function denormalize(tensor, min, max) {
  return tensor.mul(max.sub(min)).add(min);
}

// 데이터를 Min-Max 스케일링을 이용하여 정규화하는 함수
function MinMaxScaling(tensor, prevMin = null, prevMax = null) {
  // 이전값이 있으면 이전값(prevMin)을 쓰고 없으면 tensor.min() 최소값을 구한다. 
  const min = prevMin || tensor.min();
  const max = prevMax || tensor.max(); 

  // 정규화(0-1사이의 값) : 데이터샘플 - 최소값 / 최대값 - 최소값
  const normedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor : normedTensor, 
    min, 
    max
  }
}

