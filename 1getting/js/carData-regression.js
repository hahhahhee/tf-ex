// JSON 형식의 자동차 데이터를 가져온 후, 마력과 연비 정보만 추출하고 null 데이터를 제거
async function getData() {
  const carDataResponse = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json");

  const carData = await carDataResponse.json();
  console.log(carData)

  // 마력에 따른 연비계산
  const cleaned = carData.map( car => ({
    mpg : car.Miles_per_Gallon,
    horsepower : car.Horsepower
  })).filter(car => (car.mpg != null && car.horsepower != null))
  console.log(cleaned)
  return cleaned;
}
// getData()


// 모델을 생성
function createModel() {
  const model = tf.sequential()

  // 레이어 하나, 입력:inputShape 출력층:units
  // model.add(tf.layers.dense({inputShape:[1], units:1}))

  // 중간에 히든레이어를 하나 넣어주는 2-layer모델을 만들자
  // 입력에 하나, 히든층 3개, activation 추가, relu
  model.add(tf.layers.dense({inputShape:[1], units:3, activation:'relu'}));
  model.add(tf.layers.dense({units:1}))
  return model;
}

// 데이터를 섞는다 => tensor 자료형으로 변환 => 데이터 정규화
function convertToTensor(data) {
  return tf.tidy(() => {
    // step1. 데이터 셔플링
    tf.util.shuffle(data);

    // step2. 데이터를 2d tensor로 변환
    // 입력(inputs, 마력), 정답(labels, 연비)
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    // 2D tensor로 변환, 2차원 array로 바꾼다
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // step3. 0-1사이로 min-max scaling(데이터 정규화)
    // 데이터샘플 - 최소값 / 최대값 - 최소값 (0-1사이의 값)

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs : normalizedInputs,
      labels : normalizedLabels,

      // 나중에 denormalize, 원래의 값으로 바꿀때 필요함, 
      inputMax, 
      inputMin,
      labelMax,
      labelMin
    }
  })
}

// 모델을 학습시키자
// loss 손실함수 : 1 Linear Regression : meanSquareError
// metrics : 훈련되는 중간중간에 찍겠다. meanSquareError를 찍는다
async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer : tf.train.adam(), 
    loss : tf.losses.meanSquaredError, 
    metrics : ['mse']
  })

  const batchSize = 32;
  const epochs = 50;

  // 학습을 시킨다. 
  return await model.fit(inputs, labels, {
    batchSize, 
    epochs, 
    shuffle : true, 
    // 훈련과정을 시각화하기 위하여 콜백으로 지정, 에폭이 한번 끝날때마다 부르도록 지정한다.
    callbacks : tfvis.show.fitCallbacks(
      {name: 'Training Performance'}, 
      ['loss'], 
      {height: 200, callbacks: ['onEpochEnd']}
    )
  })
}

// 학습된 머신러닝 모델로 예측을 수행하고, 원본 데이터와 예측 결과를 시각화
function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMax, labelMin} = normalizationData;

  // 0-1사이로 가상의 input data 100를 생성하여 예측
  const [xs, preds] = tf.tidy(() => {

    // test data, 1차원
    const xs = tf.linspace(0, 1, 100);

    // 2차원으로 만들어서 예측(predict)시켜 preds에 넣음
    // xs.reshape([100, 1])
    const preds = model.predict(xs.reshape([100, 1]))

    // 원래 데이터로 복원, 원래값
    // 복원 x = 정규화된 값 * (최대값 - 최소값) + 최소값
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    // 예측한 값, 원래 데이터로 복원
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // dataSync() 메서드는 이 데이터를 CPU 메모리로 복사하여 JavaScript 배열로 반환
    return [unNormXs.dataSync(), unNormPreds.dataSync()]
  })

  // 시각화하기 위하여 x축과 y축의 값을 정해준다. 
  // const predictedPoints = Array.from(xs).map((val, i) => {
  //   return {
  //     x : val, 
  //     y : preds[i]
  //   }
  // })

    const predictedPoints = Array.from(xs).map((val, i) => {
      return {
        x : val, 
        y : Array.from(preds)[i]
      }
    })

  // x축은 마력, y축은 연비
  const originalPoints = inputData.map( d => ({
    x: d.horsepower, y: d.mpg
  }))

  // 시각화
  tfvis.render.scatterplot(
    {name: "Model 예측값 vs Original Data"},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: '마력',
      yLabel: '연비',
      height: 300
    }
  )


}

// 위의 과정을 순차적으로 실행해보자
async function run() {

  // 훈련할 원본 데이터
  const data = await getData()

  // 원본데이터를 이용하여 시각화해보자
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }));

  tfvis.render.scatterplot(
    {name: '마력 대 연비'}, 
    {values, series: ['original data']},
    {
      xLabel: '마력', 
      yLabel: '연비',
      height: 300
    }
  )

  // 모델 생성
  const model = createModel()
  tfvis.show.modelSummary({name : 'Model Summary'}, model)

  // 훈련할 데이터 변환한것 확인
  const tensorData = convertToTensor(data)
  const {inputs, labels} = tensorData;

  // 확인
  inputs.print()
  labels.print()

  // 모델을 훈련시키자
  await trainModel(model, inputs, labels)
  console.log('훈련완료')

  // 훈련시킨것 확인
  testModel(model, data, tensorData);

}

run()

// 페이지로드가 완료되면 run함수를 실행
document.addEventListener('DOMContentLoaded', run)