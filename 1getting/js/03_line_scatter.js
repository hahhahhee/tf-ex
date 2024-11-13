// 03_line_scatter.js

// linechart, 선형차트

// 0에서 100사이의 수를 무작위로 만들고, y값에 x값을 더하여 x, y쌍의 객체를 생성
const series1 = Array(100).fill(0).map(y => Math.random()*100).map((y, x) => ({x, y: x+y}))
console.log(series1)

// 0에서 5사이의 수를 무작위로 만들고, y값에 x값을 더하여 x, y쌍의 객체 생성
const series2 = Array(100).fill(0).map(y => Math.random()*5).map((y, x) => ({x, y: x+y}))
console.log(series2)

// 선형차트 그리기
const surface = {
  name : "Line Chart",
  tab : "선형 차트"
}

tfvis.render.linechart(surface, {
  values : [series1, series2],    // 데이터 
  series : ["big100", "small5"]   // 범례
})

// 산점도 그리기, scatterplot
const surface2 = {
  name : "Scatter Plot",
  tab : "산점도"
}

tfvis.render.scatterplot(surface2, {
  values : [series1, series2],
  series : ["big100", "small5"]
})

