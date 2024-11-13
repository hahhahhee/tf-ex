// 변수 선언
let model;          // 학습할 모델
let dataPoints;     // 로드된 데이터를 저장할 변수
let X_train, X_test, y_train, y_test;     // 학습 및 테스트 데이터
let normParams;     // 정규화를 위한 최소값, 최대값 저장

// tfvis의 visor toggle
function toggleVisor() {
  tfvis.visor().toggle();
}

// Min-Max 정규화 함수
function MinMaxScaling(tensor, prevMin = null, prevMax = null) {
  const min = prevMin || tensor.min();
  const max = prevMax || tensor.max();
  const normedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor : normedTensor, 
    min, 
    max
  }
}

function plot(dataPoints) {
  const surface = { name: "Binary Scatterplot", tab: "Data" }
  const series1 =  dataPoints.filter(v => v.y[0] == 1).map(v => ({
    x : v.x[1],    // 나이
    y : v.x[2]     // 가격(연봉)
  }))

  const series2 =  dataPoints.filter(v => v.y[0] == 0).map(v => ({
    x : v.x[1],    // 나이
    y : v.x[2]     // 가격(연봉)
  }))

  const data = { values: [series1, series2], series: ["구매", "비구매"] }

  const opts = { xLabel: "Age", yLabel: "Salary" }

  tfvis.render.scatterplot (surface, data, opts)
}

async function run() {
  const dataUrl = "Social_Network_Ads.csv"
  const socialSales = tf.data.csv(dataUrl, {
    columnNames: ["UserID", "Gender", "Age", "EstimatedSalary", "Purchased"],
    columnConfigs: {
      Gender: { isLabel : false },
      Age: { isLabel : false },
      EstimatedSalary: { isLabel : false },
      Purchased: { isLabel : true }
    }, 
    configuredColumnsOnly: true
  })

  // 10개의 배열만 출력 해보자
  console.log(await socialSales.take(10).toArray())

  // 앞의 데이터 받을 것을 보면 xs와 ys로 들어온 것을 확인할 수 있음, 시각화를 하기 위해 x와 y로 값을 넣어주자
  let dataset = await socialSales.map(({xs, ys}) => ({
    x : Object.values(xs),
    y : Object.values(ys)
  }))

  console.log(await dataset.take(10).toArray())

  // 로드된 데이터를 저장하고 섞어준다
  dataPoints = await dataset.toArray()
  tf.util.shuffle(dataPoints)

  // 문자는 숫자로 변환 : 문자는 읽을 수 없음
  // 카테고리 변수 처리 (Male -> 1, Female -> 0)
  dataPoints.map( v => {
    if(v.x[0] === "Male") {
      v.x[0] = 1;
    } else if(v.x[0] === "Female") {
      v.x[0] = 0;
    }
  })

  console.log(dataPoints)
  console.log(dataPoints.length)

  // dataPoints를 시각화 해보자, plot() 함수 구현
  // 시각화를 통하여 데이터를 파악한다. 
  plot(dataPoints);

  // 훈련을 하기 위한 준비
  const featureValue = dataPoints.map(p => p.x);    // 성별(남1,여0), 나이, 연봉
  console.log(featureValue)

  const labelValues = dataPoints.map(p => p.y);
  console.log(labelValues)

  // 텐서자료형(2차원 배열)로 바꾼다
  const featureTensor = tf.tensor2d(featureValue, [featureValue.length, 3])
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])
  featureTensor.print()
  labelTensor.print()

  // 훈련용/테스트용 데이터 셋 분리 (75% 훈련용, 25% 테스트용)
  const trainLen = Math.floor(dataPoints.length*0.75);
  const testLen = dataPoints.length - trainLen;

  [X_train, X_test] = tf.split(featureTensor, [trainLen, testLen]);
  [y_train, y_test] = tf.split(labelTensor, [trainLen, testLen]);

  console.log(X_train.shape, X_test.shape, y_train.shape, y_test.shape);
  await X_train.print()

  // 데이터 정규화 (MinMaxScaling)
  // X_train
  const normedTrainTensor = MinMaxScaling(X_train)
  normParams = { min : normedTrainTensor.mim, max : normedTrainTensor.max };
  X_train.dispose();
  X_train = normedTrainTensor.tensor;
  X_train.print();

  // X_test
  const normedTestTensor = MinMaxScaling(X_test, normParams.min, normParams.max);
  X_test.dispose();
  X_test = normedTestTensor.tensor;
  X_test.print();

  // 모델 훈련 버튼 활성화
  document.getElementById("train-button").removeAttribute("disabled");

}

async function train() {
  document.getElementById("model-status").innerHTML = "Training...";

  // sequential 모델 생성 및 설정
  model = tf.sequential();

  // 1-layer
  model.add(
    tf.layers.dense({
      inputShape:[3],   // 성별, 나이, 연봉
      units:50, 
      activation:"relu"
    })
  );

  // 2-layer
  model.add(
    tf.layers.dense({
      units:20, 
      activation:"relu"
    })
  );

  // 3-layer
  model.add(
    tf.layers.dense({
      units:1, 
      activation:"sigmoid"
    })
  );

  model.compile({
    loss: "binaryCrossentropy",  // 손실함수
    optimizer: "adam",           // 최적화 알고리즘
    metrics: ["accuracy"]        // 평가지표, accuracy는 정확도, 예측이 맞는 비율을 평가
  });

  tfvis.show.modelSummary({name: "Model Summary"}, model);

  // 모델 훈련 (fit)
  const history = await model.fit(X_train, y_train, {
    epochs: 50, 
    batchSize: 32, 
    validationData: [X_test, y_test], 
    callbacks: tfvis.show.fitCallbacks({ name: "Training Performance" }, [
      "loss", 
      "acc", 
      "val_loss", 
      "val_acc"
    ])
  });

  // 예측 버튼 활성화 및 훈련 완료 알림
  document.getElementById("predict-button").removeAttribute("disabled");
  document.getElementById("model-status").innerHTML = "Train 완료!!!, Start Prediction !!";

  // 예측한 X_test 데이터 confusion matrix 시각화를 위한 데이터 준비
  const predictions = model.predict(X_test);
  predictions.print()

  // const predClasses1 = tf.tidy(() => tf.floor(predictions.add(0.5)))
  // predClasses1.print()
  // console.log(predClasses1.shape)     // [100,1]

  // const predClasses2 = tf.tidy(() => tf.floor(predictions.add(0.5)).transpose())
  // predClasses2.print()     // [[1, 0, 0, ..., 0, 0, 1],]
  // console.log(predClasses2.shape)     // [1, 100]

  // 예측데이터
  const predClasses = tf.tidy(() => 
    tf.floor(predictions.add(0.5))
      .transpose()
      .squeeze())     // 1인 차원을 버림
  predClasses.print()     // [0, 0, 0, ..., 0, 0, 0]

  // 정답데이터
  const trueClasses = tf.tidy(() => 
    tf.floor(y_test.add(0.5))
      .transpose()
      .squeeze()
  );
  trueClasses.print()     // [0, 0, 0, ..., 0, 0, 1]

  // confusion matrix 시각화
  const confusionMatrix = await tfvis.metrics.confusionMatrix(
    trueClasses, 
    predClasses
  )
  const container = { name: "Confusion Matrix", tab: "혼동행렬" }; 
  const data = { values: confusionMatrix };
  tfvis.render.confusionMatrix(container, data)

}

function predict() {
  // 예측에 필요한 입력 값 받아오기
  const predInputOne = parseInt(document.getElementById("predict-input-1").value)
  const predInputTwo = parseInt(document.getElementById("predict-input-2").value)
  const predInputThree = parseInt(document.getElementById("predict-input-3").value)

  // 확인
  console.log(predInputOne, predInputTwo, predInputThree)

  if(isNaN(predInputOne) || isNaN(predInputTwo) || isNaN(predInputThree)) {
    alert("숫자를 입력하세요")
  } else {
    
    const features = [predInputOne, predInputTwo, predInputThree];
    // tensor 변환
    const tempTensor = tf.tensor2d(features, [1, 3]);
    // 정규화
    const normedTensor = MinMaxScaling(tempTensor, normParams.min, normParams.max);
    tempTensor.dispose();
    normedTensor.tensor.print();

    // 입력 값으로 예측 수행
    const prediction = model.predict(normedTensor.tensor)
    let prediceted;
    if (prediction.dataSync()[0] > 0.5) {
      prediceted = "구매할 고객";
    } else {
      prediceted = "구매하지 않을 고객";
    }

    // 예측 결과 출력
    document.getElementById("predict-output").innerHTML = `예측된 분류 - ${prediceted}`;

  }
}

// 웹 페이지 로딩이 완료되면 run 함수 실행
document.addEventListener("DOMContentLoaded", run);


// const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
// x.print()
// // tf.split (x, numOrSizeSplits, axis?)
// const [a, b] = tf.split(x, 2, 1)
// a.print()
// b.print()
// const [c, d, e] = tf.split(x, [1,2,1], 1)
// c.print()
// d.print()
// e.print()
// const [f, g] = tf.split(x, 2, 0)
// f.print()
// g.print()