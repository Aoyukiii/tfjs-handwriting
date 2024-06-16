const { createCanvas } = require('canvas')
const fs = require('fs')

const {
    drawFrame,
    drawRect,
    drawText,
    drawVText, 
    color
} = require('./drawBasic')

// 生成数据（示例数据）
// const data = [
//     [97, 0, 1, 2, 0],
//     [3, 97, 0, 0, 0],
//     [0, 0, 99, 0, 1],
//     [3, 0, 1, 95, 3],
//     [0, 0, 1, 0, 99],
// ]

// drawHeatMap('hmDemo', data, 100, 0)


function drawHeatMap(title, data, max, min, xLabel = 'x axis', yLabel = 'y axis') {
    // 长度参数定义
    const indent = 100
    const sqLength = 200

    const matrix = {
        widthNum: data[0].length,
        heightNum: data.length,
        width: data[0].length * sqLength,
        height: data.length * sqLength,
        x: 3 * indent,
        y: 3 * indent
    }
    
    const colorBar = {
        width: indent,
        height: matrix.height,
        x: 5 * indent + matrix.width,
        y: 3 * indent
    }

    const canva = {
        width: 7 * indent + matrix.width,
        height: 4 * indent + matrix.height,
        x: 0,
        y: 0
    }

    // 创建画布
    const canvas = createCanvas(canva.width, canva.height)
    const ctx = canvas.getContext('2d')

    // 绘制白色背景
    ctx.fillStyle = `rgb(255, 255, 255)`
    drawRect(ctx, canva)

    // 绘制黑色边框
    ctx.fillStyle = `rgb(0, 0, 0)`
    drawFrame(ctx, matrix)
    drawFrame(ctx, colorBar)

    // 绘制矩阵
    data.forEach((row, rowIndex) => {
        row.forEach((value, colIndex) => {
            const thisSquare = {
                width: sqLength,
                height: sqLength,
                x: matrix.x + colIndex * sqLength,
                y: matrix.y + rowIndex * sqLength
            }

            ctx.fillStyle = color(value / max)
            drawRect(ctx, thisSquare)

            // 在方块中间绘制对应的数字
            ctx.fillStyle = 'black'
            ctx.font = '50px Arial'
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            drawText(ctx, {
                text: value,
                x: thisSquare.x + 0.5 * sqLength,
                y: thisSquare.y + 0.5 * sqLength
            })
        })
    })

    // 绘制颜色条
    const gradient = ctx.createLinearGradient(0, colorBar.y + colorBar.height, 0, colorBar.y)
    gradient.addColorStop(0, color(0))
    gradient.addColorStop(1, color(1))
    ctx.fillStyle = gradient
    drawRect(ctx, colorBar)

    // 绘制矩阵附近文字
    ctx.fillStyle = 'black'
    ctx.font = 'bold 60px Arial'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    for (let i = 0; i < matrix.widthNum; i++) {
        drawText(ctx, {
            text: i,
            x: matrix.x + 0.5 * sqLength + i * sqLength,
            y: matrix.y - 0.4 * indent
        })
    }
    for (let i = 0; i < matrix.heightNum; i++) {
        drawText(ctx, {
            text: i,
            x: matrix.x - 0.4 * indent,
            y: matrix.y + 0.5 * sqLength + i * sqLength
        })
    }
    ctx.font = 'bold 70px Arial'
    drawText(ctx, {
        text: xLabel,
        x: matrix.x + 0.5 * matrix.width,
        y: matrix.y - 1.5 * indent
    })
    drawVText(ctx, {
        text: yLabel,
        x: matrix.x - 1.5 * indent,
        y: matrix.y + 0.5 * matrix.height
    })

    // 绘制颜色条附近文字
    ctx.font = '50px Arial'
    ctx.textAlign = 'right'
    drawText(ctx, {
        text: max,
        x: colorBar.x - 0.5 * indent,
        y: colorBar.y
    })
    drawText(ctx, {
        text: (max + min) / 2,
        x: colorBar.x - 0.5 * indent,
        y: colorBar.y + 0.5 * colorBar.height
    })
    drawText(ctx, {
        text: min,
        x: colorBar.x - 0.5 * indent,
        y: colorBar.y + colorBar.height
    })

    // 保存热力图为图像文件
    const out = fs.createWriteStream(`./training-result/${title}.png`)
    const stream = canvas.createPNGStream()
    stream.pipe(out)
    out.on('finish', () => console.log(`Heatmap has been saved as ${title}.png.`))
}

module.exports = {
    drawHeatMap
}