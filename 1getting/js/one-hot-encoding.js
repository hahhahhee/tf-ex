// category 변수의 one-hot-encoding
// x: 모델의 input, y: 정답

const myArray = [
  { x: [1, 0], y: 0 },
  { x: [5, 1], y: 1 },
  { x: [1, 1], y: 2 }
]

// label 생성
const labels = myArray.map(v => v.y)
console.log(labels)

// labels를 텐서자료형으로, 1차원 텐서
let labelTensor = tf.tensor1d(labels) 
labelTensor.print()

// 3가지를 구분하는 분류문제이므로, 3개의 열이 생김
labelTensor = tf.oneHot(labels, 3);
console.log(labelTensor.arraySync())