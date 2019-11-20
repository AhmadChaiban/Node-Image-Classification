////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Loading the model here. This is where you can test. 

//Usage: node test.js [imagePath], this will get the saved model and print the classification results

//please note again that in order to test the model the following need to be installed:
//@tensorflow/tfjs
//@tensorflow/tfjs-node@1.2.11
//@tensorflow-models/mobilenet
//node-localstorage
//fs
//@tensorflow-models/knn-classifier

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const knnClassifier = require('./node_modules/@tensorflow-models/knn-classifier/dist/knn-classifier');   //Fetching the KNN classifier directly from the directory here.

const tf = require('@tensorflow/tfjs');    //Fetching requirements
const tfnode = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
var LocalStorage = require("node-localstorage").LocalStorage;
var localStorage = new LocalStorage('./');
const fs = require('fs');

function cropImage(img) {   //Cropping the image here to make it work with mobile net
  const width = img.shape[0];  
  const height = img.shape[1];  // use the shorter side as the size to which we will crop  
  const shorterSide = Math.min(img.shape[0], img.shape[1]);  // calculate beginning and ending crop points  
  const startingHeight = (height - shorterSide) / 2;  
  const startingWidth = (width - shorterSide) / 2; 
  const endingHeight = startingHeight + shorterSide;  
  const endingWidth = startingWidth + shorterSide;  // return image data cropped to those points  
  return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
}

function loadModel(directory) {
   var classifier = knnClassifier.create();
    //can be change to other source
   let dataset = localStorage.getItem(directory);          //Using this function to load the model
   let tensorObj = JSON.parse(dataset);
   //covert back to tensor
   Object.keys(tensorObj).forEach((key) => {
     tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length/1024, 1024]); 
   })
   classifier.setClassifierDataset(tensorObj);
   return classifier;                                    //returns the classifier
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

const testModel = async path => {     //Using this function to test the model
    var model = await mobilenet.load(); 

    var classifier = await loadModel("KnnSaved");   //Loading the saved model
    
    var image = readImage(path);    //Reading the image from the specified path
    var activateTest = await model.infer(image,'conv_preds');

    var predictions = await classifier.predictClass(activateTest); //predicting classes here
     
    console.log("Classification Results: ",predictions);         //printing results here.
}

if (process.argv.length !== 3) throw new Error('Incorrect arguments: node classify.js <IMAGE_FILE>');

testModel(process.argv[2]);       //Running the model testing here       