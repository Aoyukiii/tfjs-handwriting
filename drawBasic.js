function drawRect(ctx, rect) {
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height)
}

function drawFrame(ctx, rect) {
    ctx.fillStyle = 'black'
    ctx.fillRect(rect.x - 5, rect.y - 5, rect.width + 10, rect.height + 10)
    ctx.fillStyle = 'white'
    drawRect(ctx, rect)
}

function drawText(ctx, rect) {
    ctx.fillText(rect.text, rect.x, rect.y)
}

function drawVText(ctx, rect) {
    ctx.save()
    ctx.translate(
        rect.x,
        rect.y
    )
    ctx.rotate(-Math.PI / 2)
    drawText(ctx, {
        text: rect.text,
        x: 0,
        y: 0
    })
    ctx.restore()
}

function drawSqPoint(ctx, coor) {
    const radius = 10
    ctx.fillRect(coor.x - radius, coor.y - radius, 2 * radius, 2 * radius)
}

function color(rate) {
    return `rgb(${Math.round(255 - 180 * rate)}, ${Math.round(255 - 180 * rate)}, 255)`
}

module.exports = {
    drawRect,
    drawFrame,
    drawText,
    drawVText,
    drawSqPoint,
    color
}