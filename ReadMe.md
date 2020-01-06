# Transfer Learning with Tensorflow.js:

The purpose of this project is to create an open source transfering learning model in Javascript that a user can train download
and train on any dataset of images they desire. The model that will be used in this case is the Mobile Net model from Tensorflow.js. 
This model will first extract some features from the images, then add them as a examples to the KNN classifier for training. 

A visual representation of the above logic can be seen: 

### In the classify.js file: 
	
      //Reading each image from the path
      imageSample = readImage(filesPerClass[i][j]);
      //Extracting features from the model
      activation = await model.infer(imageSample, 'conv_preds');
      //giving some feedback to the user for progress
      console.log('Training Class '+folders[i]+' image ' + (j+1) +" "+filesPerClass[i][j].split("/")[3]);
      //Training the model here by addExample. Label is being set as the folder name
      classifier.addExample(activation,folders[i]);

It takes the folder name of the images and uses that as the label for that specific category of images. The classifier
can then be saved through the following function located in classify.js:

     function saveModel(classifier,directory){
       let dataset = classifier.getClassifierDataset()
       var datasetObj = {}
       Object.keys(dataset).forEach((key) => {
         let data = dataset[key].dataSync();
         // use Array.from() so when JSON.stringify() it coverts to an array string e.g [0.1,-0.2...] 
         // instead of object e.g {0:"0.1", 1:"-0.2"...}
         datasetObj[key] = Array.from(data); 
       });
       let jsonStr = JSON.stringify(datasetObj);
       //can be change to other source
       localStorage.setItem(directory, jsonStr);
     }

The user can then use the model to classify any image they want in the test.js file. The mobile net model, as well as the saved KNN model
are reloaded and after the mobile net model extracts some features, the saved KNN model can then classify the image. 

### In test.js:

     //Using this function to test the model
     const testModel = async path => { 
         // Loading the Mobile Net model
         var model = await mobilenet.load(); 
         // Loading the Saved Model
         var classifier = await loadModel("KnnSaved");
         //Reading the image from the specified path
         var image = readImage(path); 
         //Applying a feature extraction to the image 
         var activateTest = await model.infer(image,'conv_preds');
         //Using the KNN model to classify the image
         var predictions = await classifier.predictClass(activateTest);
         //Printing the classification results
         console.log("Classification Results: ",predictions); 
     }

In this case I have only allowed testing for single images. One can, however, with a simple tweak in the code, allow for testing on multiple
images at a time. 

Giving reference to @tejas77 https://github.com/tejas77/node-image-classification for giving me a starting point with his repo! Thank you!