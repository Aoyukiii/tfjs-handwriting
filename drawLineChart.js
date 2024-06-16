const { createCanvas } = require('canvas')
const fs = require('fs')

const {
    drawFrame,
    drawRect,
    drawText,
    drawVText, 
    color,
    drawSqPoint
} = require('./drawBasic')
const { mod } = require('@tensorflow/tfjs')

// 测试数据
// const arr = [0, 40, 70, 85, 92, 94, 97, 96, 98, 96, 99, 100]

// drawLineChart('lcDemo', arr)

function drawLineChart(title, data, max, min, xTickSpan = 5, yTickNum = 5, lineColor = 'blue') {
    if (max === undefined) max = Math.max(...data)
    if (min === undefined) min = Math.min(...data)

    // 长度参数定义
    const indent = 100
    const chartSpace = 50
    const chartInnerHeight = 800
    const pointSpace = 100

    const dataNum = data.length
    const chartInnerWidth = (dataNum - 1) * pointSpace

    const canva = {
        width: chartInnerWidth + 3 * indent + 2 * chartSpace,
        height: chartInnerHeight + 4 * indent + 2 * chartSpace,
        x: 0,
        y: 0
    }

    const chartInner = {
        width: chartInnerWidth,
        height: chartInnerHeight,
        x: 2 * indent + chartSpace,
        y: 2 * indent + chartSpace
    }

    const chart = {
        width: chartInner.width + 2 * chartSpace,
        height: chartInner.height + 2 * chartSpace,
        x: 2 * indent,
        y: 2 * indent
    }

    // 创建一个 Canvas 实例
    const canvas = createCanvas(canva.width, canva.height)
    const ctx = canvas.getContext('2d')

    // 绘制白色背景
    ctx.fillStyle = 'rgb(255, 255, 255)'
    drawRect(ctx, canva)
    
    // 绘制标题
    ctx.fillStyle = 'black'
    ctx.font = 'bold 70px Arial'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    drawText(ctx, {
        text: title,
        x: 0.5 * canvas.width,
        y: indent
    })

    // 绘制折线图边框
    drawFrame(ctx, chart)

    // 绘制折线图刻度


    // 绘制折线图
    ctx.beginPath()
    ctx.fillStyle = lineColor
    ctx.font = '40px Arial'

    for (let i = 0; i < dataNum; i++) {
        const thisCoor = {
            x: chartInner.x + i * chartInner.width / (dataNum - 1),
            y: chartInner.y + chartInner.height - chartInner.height * (data[i] - min) / (max - min)
        }

        ctx.fillStyle = lineColor
        drawSqPoint(ctx, thisCoor)

        drawTick(ctx, {
            x: thisCoor.x,
            y: chart.y + chart.height
        })

        if (i % xTickSpan === xTickSpan - 1) {
            drawText(ctx, {
                text: i + 1,
                x: thisCoor.x,
                y: chart.y + chart.height + 0.5 * indent
            })
        }

        if (i === 0) ctx.moveTo(thisCoor.x, thisCoor.y)
        else         ctx.lineTo(thisCoor.x, thisCoor.y)
    }
    ctx.lineWidth = 3
    ctx.strokeStyle = lineColor
    ctx.stroke()

    for (let i = 0; i < yTickNum; i++) {
        
        drawVTick(ctx, {
            x: chart.x,
            y: chartInner.y + chartInner.height - (i / (yTickNum - 1)) * chartInner.height
        })

        const value = min + (i / (yTickNum - 1)) * (max - min)
        ctx.textAlign = 'right'
        drawText(ctx, {
            text: value.toFixed(4),
            x: chart.x - 0.3 * indent,
            y: chartInner.y + chartInner.height - (i / (yTickNum - 1)) * chartInner.height
        })
    }


    // 保存热力图为图像文件
    const out = fs.createWriteStream(`./training-result/${title}.png`)
    const stream = canvas.createPNGStream()
    stream.pipe(out)
    out.on('finish', () => console.log(`Line chart has been saved as ${title}.png.`))
}

function drawTick (ctx, coor) {
    ctx.fillStyle = 'black'
    const width = 2
    const height = 15
    drawRect(ctx, {
        x: coor.x - 0.5 * width,
        y: coor.y - height,
        width,
        height
    })
}

function drawVTick (ctx, coor) {
    ctx.fillStyle = 'black'
    const width = 15
    const height = 2
    drawRect(ctx, {
        x: coor.x,
        y: coor.y - 0.5 * height,
        width,
        height
    })
}

module.exports = {
    drawLineChart
}