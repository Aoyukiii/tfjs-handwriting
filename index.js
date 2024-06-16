const tf = require('@tensorflow/tfjs')
const mnist = require('mnist')
const fs = require('fs')

const { drawHeatMap } = require('./drawHeatMap')
const { drawLineChart } = require('./drawLineChart')

/* =========================================================================== */

const trainNum = 7740
const validateNum = 860
const testNum = 8000

const epochs = 20
const batchSize = 256

// 准备数据
const trainSet = mnist.set(trainNum, validateNum)
const testSet = mnist.set(0, testNum)

// 训练集和验证集集导入
const trainInputs      = tf.tensor(trainSet.training.map(obj => obj.input ))
const trainOutputs     = tf.tensor(trainSet.training.map(obj => obj.output))
const validateInputs   = tf.tensor(trainSet.test    .map(obj => obj.input ))
const validateOutputs  = tf.tensor(trainSet.test    .map(obj => obj.output))

// 测试集导入
const testInputs = tf.tensor(testSet.test.map(obj => obj.input))
const testOutputs = tf.tensor(testSet.test.map(obj => obj.output))
/* =========================================================================== */

// 模型参数设置
const regularizer = tf.regularizers.l2({l2: 0.001})
const model = tf.sequential({
    layers: [
        tf.layers.dense({inputShape: 784, units: 256, kernelRegularizer: regularizer, activation: 'relu'}),
        tf.layers.dense({units: 256, activation: 'relu'}),
        tf.layers.dense({units: 256, activation: 'relu'}),
        tf.layers.dense({units: 256, activation: 'relu'}),
        tf.layers.dense({units: 10, activation: 'softmax'}),
    ]
})

// 指定优化器、损失函数和指标列表
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
})

/* =========================================================================== */

// 模型训练函数
function train() {
    return new Promise(trainPromise)
}

function trainPromise(resolve) {
    model.fit(trainInputs, trainOutputs, {
        epochs,
        batchSize,
        validationData: [validateInputs, validateOutputs],
        callbacks: {
            onEpochEnd
        }
    })
    .then(info => {
        // fit执行完返回的info对象是包含训练过程产生的数据
        // 这里我们把info.history对象的内容以json格式存到文件中
        const history = JSON.stringify(info.history, null, 4)
    
        fs.writeFile('./training-result/history.json', history, err => {
            if (err) {
                console.error('writeFileError:', err)
            }
            console.log('File has been written successfully.')
            resolve(0)
        })

        // 绘制折线图
        drawLineChart('Accuracy', info.history.val_acc, 1)
        drawLineChart('Loss', info.history.val_loss)
    })
}

// 训练过程的回调函数
function onEpochEnd(epochs, info) {
    console.log(`#${epochs + 1}`)
    console.log(`├── Acc: ${(info.acc * 100).toFixed(2)} %`)
    console.log(`├── ValAcc: ${(info.val_acc * 100).toFixed(2)} %`)
    console.log(`├── Loss: ${info.loss.toFixed(2)}`)
    console.log(`└── ValLoss: ${info.val_loss.toFixed(2)}`)
}

/* =========================================================================== */

// 模型测试
function test() {
    // 使用热图进行绘制，更加直观
    // 热图的第一维度表示标签，第二维度表示该标签的推测值
    const HeatMatrix = Array.from(Array(10), () => Array(10).fill(0))

    // predictions和labels都是number[]类型
    const labels      = getAns(testOutputs)
    const predictions = getAns(model.predict(testInputs))

    for (let i = 0; i < labels.length; i++) {
        const label      = labels     [i]
        const prediction = predictions[i]
    
        HeatMatrix[label][prediction]++
    }

    // 绘制热图
    drawHeatMap(
        'testHeatmap',
        HeatMatrix,
        testNum / 10,
        0,
        'Predictions',
        'Labels'
    )
}

// 把tf.Tensor[]类型转化为number[]类型
function getAns(origin) {
    const tensorArr = tf.split(
        origin,
        testOutputs.shape[0],
        0
    )

    const resultArr = Array.from(tensorArr, (tensor) => {
        return tensor.transpose().argMax().dataSync()[0]
    })

    return resultArr
}

/* =========================================================================== */

// 开始训练
train()
.then(test)
.catch(err => {
    console.error('Error:', err)
})