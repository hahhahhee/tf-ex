// 01_barchart.js

// bar chart

// data 생성
const data = [
  {index : 0, value : 50},
  {index : 1, value : 100},
  {index : 2, value : 150}
]

// bar chart 그리기
const surface = {
  name : 'Bar chart',
  tab : 'Chart1'
}

// tfvis.render.barchart(surface, data, opts?);
tfvis.render.barchart(surface, data);

// bar차트 응용, 옵션 추가
// 1부터 5까지 인덱스와 그에 따라 증가하는 값을 가진 데이터를 생성
const data1 = []
for (let i=1; i<=5; i++) {
  data1.push({index:i, value:i*50})
}

// bar차트 그리기
const surface1 = {name : 'Bar chart', tab : 'Chart2'}

tfvis.render.barchart(surface1, data1, {
  xLabel : 'x-축',
  yLabel : 'y-축'
})