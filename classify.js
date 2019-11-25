/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Custom Transferred model. This model takes a main directory (that contains folders that contain images). It then searches for 
//folders within that contain images and then trains the images. It specifies the classes by the number of 
//folders contained in the main directory. 

//Usage: node classify.js in terminal I have removed the image input from the terminal for testing, it is 
//now specified in the bottom. Where the prediction is being made
//have fun!   

//Please note that after npm install, KNN-classifier is required to be installed manually, as well as 
//@tensorflow/tfjs-node@1.2.11 which is the required version and node-localstorage requires installation.

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const tf = require('@tensorflow/tfjs');
//MobileNet : pre-trained model for TensorFlow.js
const mobilenet = require('@tensorflow-models/mobilenet');
//The module provides native TensorFlow execution
//in backend JavaScript applications under the Node.js runtime.
const tfnode = require('@tensorflow/tfjs-node');

const knnClassifier = require('./node_modules/@tensorflow-models/knn-classifier/dist/knn-classifier');   //Fetching the KNN classifier directly from the directory here.

//The fs module provides an API for interacting with the file system.
const fs = require('fs');

var LocalStorage = require("node-localstorage").LocalStorage;
var localStorage = new LocalStorage('./');

function cropImage(img) {  //Cropping the image here to make it work with mobile net
  const width = img.shape[0];  
  const height = img.shape[1];  // use the shorter side as the size to which we will crop  
  const shorterSide = Math.min(img.shape[0], img.shape[1]);  // calculate beginning and ending crop points  
  const startingHeight = (height - shorterSide) / 2;  
  const startingWidth = (width - shorterSide) / 2; 
  const endingHeight = startingHeight + shorterSide;  
  const endingWidth = startingWidth + shorterSide;  // return image data cropped to those points  
  return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
}

const readImage = path => {
  //reads the entire contents of a file.
  //readFileSync() is synchronous and blocks execution until finished.
  const imageBuffer = fs.readFileSync(path);
  //Given the encoded bytes of an image,
  //it returns a 3D or 4D tensor of the decoded image. Supports BMP, GIF, JPEG and PNG formats.
  var tfimage = tfnode.node.decodeImage(imageBuffer);

  var croppedImg = cropImage(tfimage);
  
  const smalImg = tf.image.resizeBilinear(croppedImg, [224, 224]); //Please not that for both cases, depending on the set of images, this size can be changed.
 
  //All images being trained are being resized to a custom size, which can be changed to what is suitable.

  const resized = tf.cast(smalImg, 'float32'); //casting the image to float32
  const t3d = tf.tensor3d(Array.from(resized.dataSync()),[224,224,3]);  //Creating a 3d tensor for the classifier
  return t3d;
}

var mainDirectory = "./img_samples/"; //Main directory that allows for the specification of the main folder of images

const imageClassification = async path => {   
  
  // Loading the trained model.
  const model = await mobilenet.load();  //loading the model here
  
  // print results on terminal
  var folders = fs.readdirSync(mainDirectory).filter(function(x){
    return x.toLowerCase() != '.ds_store' && x.toLowerCase() != 'ignore-other-file.js';
  });   //reading from the main directory.
  
  var filesPerClass = [];
  for(var i=0;i<folders.length;i++){
    files = fs.readdirSync(mainDirectory+folders[i]).filter(function(x){
      return x.toLowerCase() != '.ds_store' && x.toLowerCase() != 'ignore-other-files.js';
    });
    var files_complete = [];
    for(var j=0;j<files.length;j++){
      files_complete.push(mainDirectory+folders[i]+"/"+files[j]); //Getting all the image files here from each folder here.
    }
    filesPerClass.push(files_complete); //Using this method to separate the classes by array
  }

  for(var i=0;i<filesPerClass.length;i++){
    for(var j=0;j<filesPerClass[i].length;j++){
      imageSample = readImage(filesPerClass[i][j]); //Reading each image from the path
      activation = await model.infer(imageSample, 'conv_preds'); //Feature extraction
      console.log('Training Class '+folders[i]+' image ' + (j+1) +" "+filesPerClass[i][j].split("/")[3]); //giving some feedback to the user for progress
      classifier.addExample(activation,folders[i]);                        //Training the model here by addExample. Labels is 
    }                                                                      //being set as folder name.
  }  
  saveModel(classifier,"KnnSaved");   //Saving the model using the model name.
}

function saveModel(classifier,directory){
  let dataset = classifier.getClassifierDataset()
  var datasetObj = {}
  Object.keys(dataset).forEach((key) => {
    let data = dataset[key].dataSync();
    // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...] 
    // instead of object e.g {0:"0.1", 1:"-0.2"...}
    datasetObj[key] = Array.from(data); 
  });
  let jsonStr = JSON.stringify(datasetObj);
  //can be change to other source
  localStorage.setItem(directory, jsonStr);
}

const classifier = knnClassifier.create(); //Creating a KNN classifier here        
imageClassification();     //Running the function here. 