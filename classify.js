//TensorFlow.js is an open-source hardware-accelerated JavaScript library
//for training and deploying machine learning models.

const tf = require('@tensorflow/tfjs');
//MobileNet : pre-trained model for TensorFlow.js
const mobilenet = require('@tensorflow-models/mobilenet');
//The module provides native TensorFlow execution
//in backend JavaScript applications under the Node.js runtime.
const tfnode = require('@tensorflow/tfjs-node');

const knnClassifier = require('./node_modules/@tensorflow-models/knn-classifier/dist/knn-classifier');

//The fs module provides an API for interacting with the file system.
const fs = require('fs');

const readImage = path => {
  //reads the entire contents of a file.
  //readFileSync() is synchronous and blocks execution until finished.
  const imageBuffer = fs.readFileSync(path);
  //Given the encoded bytes of an image,
  //it returns a 3D or 4D tensor of the decoded image. Supports BMP, GIF, JPEG and PNG formats.
  var tfimage = tfnode.node.decodeImage(imageBuffer);
  
  const smalImg = tf.image.resizeBilinear(tfimage, [368, 432]); 
  //All images being trained are being resized to a custom size. 

  const resized = tf.cast(smalImg, 'float32'); //casting the image to float32
  const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,368,432,3]);  //Creating a 4d tensor for the classifier
  return t4d;
}


var mainDirectory = "./img_samples/";    //Main directory that allows for the specification of the main folder of images

const imageClassification = async path => {

  const classifier = await knnClassifier.create();    //Creating a KNN classifier here           
  
  // Loading the trained model.
  const model = await mobilenet.load();
  
  // print results on terminal
  var folders = fs.readdirSync(mainDirectory);
  
  var filesPerClass = [];
  for(var i=0;i<folders.length;i++){
    files = fs.readdirSync(mainDirectory+folders[i]);
    var files_complete = [];
    for(var j=0;j<files.length;j++){
      files_complete.push(mainDirectory+folders[i]+"/"+files[j]);
    }
    filesPerClass.push(files_complete);
  }

  for(var i=0;i<filesPerClass.length;i++){
    for(var j=0;j<filesPerClass[i].length;j++){
      imageSample = readImage(filesPerClass[i][j]);
      activation = await model.infer(imageSample, 'conv_preds');
      console.log('Training Class '+i+' image ' + (j+1))  
      classifier.addExample(activation,i);
    }
  }  
  const image = await readImage('./rabbit.jpg');   //image entered through terminal
  var activate2 = await model.infer(image,'conv_preds'); //inferring from the model (feature extraction)

  const predictionsTest = await classifier.predictClass(activate2);   
  console.log('classficationTest:',predictionsTest);
}

// if (process.argv.length !== 3) throw new Error('Incorrect arguments: node classify.js <IMAGE_FILE>');

imageClassification();