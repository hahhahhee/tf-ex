// 글로벌 변수 선언
let X_train, X_test, y_train, y_test;
let model;
let labelNames = ["setosa", "versicolor", "virginica"]

// 함수만들기
async function run() {
  // 데이터 로드
  let iris = await fetch("iris.json")
  // json형식으로 변환 => 확인
  iris = await iris.json();
  console.log(iris)
  // x축은 입력데이터, y축은 정답데이터
  let dataPoints = await iris.map(x => ({
    x: [x.sepal_length, x.sepal_width, x.petal_length, x.petal_width],
    y: x.species
  }))
  console.log(dataPoints)
  // shuffle을 위해 TensorFlow데이터셋으로 변환후 shuffle
  // 방법1: 데이터배열을 직접 섞기(이전예제에서 사용한 방법)
  // tf.util.shuffle(dataPoints)

  // 방법2: TensorFlow 데이터셋 API를 사용하여 섞기
  let dataset = tf.data.array(dataPoints);
  dataset = dataset.shuffle(dataPoints.length);
  dataPoints = await dataset.toArray();
  console.log(dataPoints)

  // data시각화
  plot(dataPoints)
}

async function plot(dataPoints) {

  const surface1 = document.querySelector("#plot1");
  const surface2 = document.querySelector("#plot2");

  // 제목 추가
  document.getElementById('plot1_title').innerText = "꽃받침 길이 & 꽃잎 길이"
  document.getElementById('plot2_title').innerText = "꽃받침 너비 & 꽃잎 너비"

  // setosa, versicolor, virginica
  // [x.sepal_length, x.sepal_width, x.petal_length, x.petal_width]
  const series1 = dataPoints.filter(v => v.y === "setosa").map(v => ({
    x: v.x[0],    // sepal_length
    y: v.x[2]     // petal_length
  }))

  const series2 = dataPoints.filter(v => v.y === "versicolor").map(v => ({
    x: v.x[0],    // sepal_length
    y: v.x[2]     // petal_length
  }))

  const series3 = dataPoints.filter(v => v.y === "virginica").map(v => ({
    x: v.x[0],    // sepal_length
    y: v.x[2]     // petal_length
  }))

  const series11 = dataPoints.filter(v => v.y === "setosa").map(v => ({
    x: v.x[1],    // x.sepal_width
    y: v.x[3]     // x.petal_width
  }))

  const series21 = dataPoints.filter(v => v.y === "versicolor").map(v => ({
    x: v.x[1],    // x.sepal_width
    y: v.x[3]     // x.petal_width
  }))

  const series31 = dataPoints.filter(v => v.y === "virginica").map(v => ({
    x: v.x[1],    // x.sepal_width
    y: v.x[3]     // x.petal_width
  }))


  // values는 data, series는 범례
  const data1 = {
    values: [series1, series2, series3], series: labelNames
  }
  const data2 = {
    values: [series11, series21, series31], series: labelNames
  }

  const opts1 = { xLabel: "sepal_length", yLabel: "petal_length" }
  const opts2 = { xLabel: "sepal_width", yLabel: "petal_width" }

  tfvis.render.scatterplot(surface1, data1, opts1);
  tfvis.render.scatterplot(surface2, data2, opts2);

  run1(dataPoints);
}

async function run1(dataPoints) {
  document.getElementById('hr').innerHTML = "<hr>";

  // dataPoints를 찍어보면 y 데이터가 문자이므로 0,1,2 숫자로 변환한다
  dataPoints.map(v => {
    if (v.y === "setosa") {
      v.y = 0;
    } else if (v.y === "versicolor") {
      v.y = 1;
    } else if (v.y === "virginica") {
      v.y = 2;
    }
  });

  console.log(dataPoints)

  // 지금은 x데이터와 y데이터가 같이 있는데 tensor로 사용하기 위해 분리한다. 
  const featureValues = dataPoints.map(v => v.x);   // feature : 훈련
  console.log(featureValues)

  const labelValues = dataPoints.map(v => v.y)      // label : 정답
  console.log(labelValues)

  // X의 shape은 [147,4]이므로 
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 4]);
  featureTensor.print(true)

  // y는 데이터가 1개 이므로 labelValues, int32 : 정수이므로
  let labelTensor = tf.tensor1d(labelValues, "int32");
  labelTensor.print(true)

  // label을 one-hot-encoding
  labelTensor = tf.oneHot(labelTensor, 3);
  labelTensor.print()

  // featureTensor를 보면 데이터값이 차이가 크지 않으므로 MinMaxScaling을 하지 않아도 된다. 

  // train test split   75/25로 나눔
  const trainLen = Math.floor(featureTensor.shape[0] * 0.75);
  const testLen = featureTensor.shape[0] - trainLen;

  // 나눈다 
  [X_train, X_test] = tf.split(featureTensor, [trainLen, testLen]);
  [y_train, y_test] = tf.split(labelTensor, [trainLen, testLen]);

  X_train.print(true)
  X_test.print(true)
  y_train.print(true)
  y_test.print(true)

  document.getElementById("train-button").removeAttribute("disabled");
}

async function train() {
  document.getElementById("model-status").innerHTML = `훈련중....`;

    // sequential 모델 생성 및 설정
    model = tf.sequential();

    // 1-layer
    model.add(
      tf.layers.dense({
        inputShape:[4],   
        units:100, 
        activation:"relu"     // 4*100+100=500
      })
    );
  
    // 2-layer
    model.add(
      tf.layers.dense({
        units:50, 
        activation:"relu"     // 100*50+50=5050
      })
    );
  
    // 3-layer
    model.add(
      tf.layers.dense({
        units:3, 
        activation:"softmax"   // 50*3+3=153
      })
    );
  
    model.compile({
      loss: "categoricalCrossentropy",  
      optimizer: "adam",           // 최적화 알고리즘
      metrics: ["accuracy"]        // 평가지표, accuracy는 정확도, 예측이 맞는 비율을 평가
    });
  

    // summary 시각화
    const surface3 = document.querySelector("#summary");
    tfvis.show.modelSummary(surface3, model);

    const surface4 = document.querySelector("#fitCallback");
    const history = await model.fit(X_train, y_train, {
      epochs: 30, 
      batchSize: 32, 
      validationData: [X_test, y_test], 
      callbacks: tfvis.show.fitCallbacks(surface4, [
        "loss", 
        "acc", 
        "val_loss", 
        "val_acc"
      ])
    });

    document.getElementById("predict-button").removeAttribute("disabled");
    document.getElementById("model-status").innerHTML = "학습 종료. 예측 시작!!";

    // 어떤 종을 잘 맞추는지 예측해 보자
    // confusion maxtrix 시각화
    const pred = model.predict(X_test);
    pred.print()    // 예측
    y_test.print()  // 정답

    // 비교
    pred.argMax(1).print()
    y_test.argMax(1).print()

    // 행(axis = 0) ↓ 또는 열(axis = 1) → 을 따라 가장 큰 값의 색인을 찾는다. 기본적으로 가장 큰 값의 인덱스는 배열을 평면화하여 찾는다. 
    // 반환 : 전체 배열에서 가장 높은 값을 가진 요소의 인덱스 반환

    const trueValue = tf.tidy(() => y_test.argMax(1));
    const predValue = tf.tidy(() => pred.argMax(1));

    const confusionMatrix = await tfvis.metrics.confusionMatrix(
      trueValue,
      predValue
    );

    document.getElementById('matrixTitle').innerHTML = "<h3>confusion matrix 시각화</h3><p>테스트 데이터를 이용하여 학습한 후 예측한 정답률을 matrix로 표현함</p>"
    let container = document.querySelector("#matrix");
    tfvis.render.confusionMatrix(container, { values: confusionMatrix });

    // per class accuracy 시각화
    // 클래스 별로 얼마나 정확하게 맞추는지
    const classAccuracy = await tfvis.metrics.perClassAccuracy(
      trueValue, 
      predValue
    )
    console.log(classAccuracy)
    let container1 = document.querySelector("#classAccuracy");
    tfvis.show.perClassAccuracy(container1, classAccuracy)

    // memory 정리
    pred.dispose()
    trueValue.dispose()
    predValue.dispose()
}

async function predict() {
  const inputOne = parseInt(document.getElementById("predict-input-1").value)
  const inputTwo = parseInt(document.getElementById("predict-input-2").value)
  const inputThree = parseInt(document.getElementById("predict-input-3").value)
  const inputFour = parseInt(document.getElementById("predict-input-4").value)

  if(isNaN(inputOne) || isNaN(inputTwo) || isNaN(inputThree) || isNaN(inputFour)) {
    alert("숫자만 입력 가능합니다.")
  } else {
    const features = [inputOne, inputTwo, inputThree, inputFour]
    console.log(features)     // [5, 5, 5, 2]
    const featureTensor = tf.tensor2d(features, [1, 4])
    featureTensor.print()     // [[5, 5, 5, 2],]
    const prediction = model.predict(featureTensor);
    prediction.print()        // [[0.0288468, 0.5881221, 0.3830312],]
    const idx = prediction.argMax(1).dataSync();
    console.log(idx)
    document.getElementById("predict-output").innerHTML = `예상되는 iris종은 ${labelNames[idx]}입니다`;
  }
}
