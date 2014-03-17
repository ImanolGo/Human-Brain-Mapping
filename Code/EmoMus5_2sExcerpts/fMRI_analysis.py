import nibabel as nib
import numpy as np
import cPickle as pickle
import os, glob
from scipy import stats,sqrt,signal,io, polyfit, spatial,linspace,polyval
import pylab
from sklearn import svm,linear_model
import xlrd, xlwt
from xlutils.copy import copy
import warnings
import random

class fMRI_analysis():
    
    def __init__(self,path,user):
        print "EmoMus5_Test_2s -> Starting fMRI analysis from user : ", user
        warnings.filterwarnings("ignore")
        self.path = path
        self.user = user
        self.m = []
        self.data = []
        self.Ts = 2
        self.numScansExcerpt = 15

    


    def readData(self):
        # Reads all the *.nii files and save it into a 4D Matrix (data)
        if os.path.exists(self.path):
           
            print "EmoMus5_Test_2s -> Loading fMRI images from user: ", self.user
            path = self.path + "/Data/" + self.user + ".nii.gz"
            print "EmoMus5_Test_2s -> Loading data from: " + path
            print "EmoMus5_Test_2s -> Loading..." 

            img = nib.load(path)
            self.data = img.get_data() #filling the data with every frame
            print "EmoMus5_Test_2s -> Data matrix with dimensions: " + str(self.data.shape)          
        else:
            print "EmoMus5_Test_2s -> User  " +  str(self.user) + " does not exist"
        
    def selectFeatures(self, orderScans):
        print "\nEmoMus5_Test_2s -> Select Features"
        self.orderScans = orderScans
        self._readEmotionGroupIndexes()
        self._dimReduction()
        #self._SVM_Analysis()
        self._saveSelectedFeatures()
        return self.f_ind, self.emotions

    def predictionModel(self,ANOVA_Ind, emotionsIndexes):

        print "\nEmoMus5_Test_2s -> Starting processing prediction model for user ", self.user

        self.f_ind = ANOVA_Ind # ANOVA analysis best voxels indexes
        self.emotions = emotionsIndexes
        self._readEmotionGroupScans() #Read emotion group scans and save it into numpy array
        self._PMAnalysis()  #Start with the prediction model analysis




    def _readEmotionGroupIndexes(self):
        
        print "EmoMus5_Test_2s -> Read music emotion group indexes "

        (self.lenX,self.lenY,self.lenZ,self.scans) = np.shape(self.data) # 4D axis
        emotion_happy = []     #Index of files belonging to happy emotional music
        emotion_anxious= []    #Index of files belonging to anxous emotional music
        emotion_neutral = [] #Index of files belonging to fear emotional music
        
        # Reads which scans belongs to which tasks 
        
        for taskNum, listScans in enumerate(self.orderScans):
          #print taskNum
          #print listScans

          if taskNum == 0: #Anxious
            for scan in listScans:
              for n in range(self.numScansExcerpt): 
                volumeInd = scan + n
                emotion_anxious.append(volumeInd) 
              

          elif taskNum == 1: #Happy
            for scan in listScans:
              for n in range(self.numScansExcerpt):
                volumeInd = scan + n
                emotion_happy.append(volumeInd) 
                  

          elif taskNum == 2: #Neutral
            for scan in listScans:
              for n in range(self.numScansExcerpt):
                volumeInd = scan + n
                emotion_neutral.append(volumeInd) 
        
        self.emotions = []
        self.emotions.append(emotion_anxious)
        self.emotions.append(emotion_happy)
        self.emotions.append(emotion_neutral) 

        print "EmoMus5_Test_2s -> Emotion: Anxious " + str(self.emotions[0])
        print "EmoMus5_Test_2s -> Emotion: Happy " + str(self.emotions[1])
        print "EmoMus5_Test_2s -> Emotion: Neutral " + str(self.emotions[2])  

    def _readEmotionGroupScans(self):
        
        print "EmoMus5_Test_2s -> Read music emotion group scans "
        
        self.happyVolumes = [] #Mean Volume for every stimuli in happy music
        self.neutralVolumes = [] #Mean Volume for every stimuli in neutral music
        self.anxiousVolumes = [] #Mean Volume for every stimuli in anxious music
       
        # Reads which scans belongs to which tasks 
        
        for emotionNum, emotion in enumerate(self.emotions):

          if emotionNum == 0: #Anxious
            print "EmoMus5_Test_2s -> Emotion: Anxious Reading " + str(len(emotion))+ " fMRI scans "
            for volumeInd in emotion:
              self.anxiousVolumes.append(self.data[:,:,:,volumeInd]) #we append the segment of every stimuli

          elif emotionNum == 1: #Happy
            print "EmoMus5_Test_2s -> Emotion: Happy Reading " + str(len(emotion))+ " fMRI scans "
            for volumeInd in emotion:
              self.happyVolumes.append(self.data[:,:,:,volumeInd]) #we append the segment of every stimuli

          elif emotionNum == 2: #Neutral
            print "EmoMus5_Test_2s -> Emotion: Neutral Reading " + str(len(emotion))+ " fMRI scans "
            for volumeInd in emotion:
              self.neutralVolumes.append(self.data[:,:,:,volumeInd]) #we append the segment of every stimuli

        print "EmoMus5_Test_2s -> Converting to numpy array... " 
        self.anxiousVolumes = np.array(self.anxiousVolumes)
        self.happyVolumes = np.array(self.happyVolumes)
        self.neutralVolumes = np.array(self.neutralVolumes)

        print "EmoMus5_Test_2s -> Emotions: Every music emotion group as a matrix with shape " + str(self.anxiousVolumes.shape)
        




    def _PMAnalysis(self):
       
      print "\nEmoMus5_Test_2s -> Starting Prediction Model Analysis"
  
      self._createDescriptorMatrices()
      self._PMAnalysis_mainLoop()

    def _PMAnalysis_mainLoop(self): 

      print "EmoMus5_Test_2s -> Prediction Model Analysis main loop"

      self._setUpMainLoopLists()

      self.test_List = ['Anxious','Happy','Neutral']
      for dist in self.distance_list:   
         print "Dist = ", dist
         indDist = self.distance_list.index(dist)
         self.ws.write(2+(len(self.test_List)+3)*indDist, 0, dist)
         for n, test in enumerate(self.test_List):
             self.ws.write(n+3+(len(self.test_List)+3)*indDist, 0, test)
         
         self.ws.write(len(self.test_List)+3+(len(self.test_List)+3)*indDist, 0, 'Total')
         for PM_method in self.PM_list:
             print "Prediction Model technique = ", PM_method
             indPM = self.PM_list.index(PM_method)
             self.ws.write(1+(len(self.test_List)+3)*indDist, 1+len(self.numFeat_list)*indPM, PM_method)
             for numFeat in self.numFeat_list:    
                 self.numSelFeatures = numFeat
                 print "NumFeat = ", numFeat   
                 self._CalcPredictionModel(PM_method)
                 self._writePM_Results(PM_method,dist)

      self.wb.save(self.path + '/ResultsPM_EmoMus5.xls')  
        
    def _setUpMainLoopLists(self): 

      #self.PM_list = ['LinearReg', 'SVR_linear' ,'SVR_poly']
      #self.PM_list = ['LinearReg','SVR_rbf']
      
      # The list of prediction model analysis methods
      self.PM_list = ['LinearReg'] 
      print "EmoMus5_Test_2s -> Prediction Model Analysis methods: " + str(self.PM_list)  

      # The list of distance types measurements to use
      self.distance_list = ['CosineDist'] 
      print "EmoMus5_Test_2s -> Prediction Model Analysis distance measurment types: " + str(self.distance_list) 
      
      # The number of features that you will use
      self.numFeat_list = [self.f_ind.shape[0]]
      print "EmoMus5_Test_2s -> Making the analysis for the best " + self.numFeat_list + " ANOVA analysis voxels"

    def _createDescriptorMatrices(self):
        
        print "EmoMus5_Test_2s -> Creating Descriptors Matrices"

        self._readDescrtiptorsFile()
        self._saveDescriptorsIntoMatrices()
        self._createLeave2OutCrossValidationIndexes()

    def _readDescrtiptorsFile(self):

        rb = xlrd.open_workbook(self.path+ '/ResultsPM_EmoMus5.xls')
        descriptorsPath = self.path+'/Info/Class_descriptors_Excerpts_2s.xls'
        print "EmoMus5_Test_2s -> Reading descriptors from " + descriptorsPath

        self.wb = copy(rb)
        self.ws = self.wb.add_sheet(self.user)
        self.ws.write(0, 0, self.user)
        
       

        wb = xlrd.open_workbook(descriptorsPath)
        self.sh = wb.sheet_by_index(0) #get the first sheet

        self.descriptorsAnxious = [] #audio descriptors for every sound file in Anxious
        self.descriptorsHappy = [] #audio descriptors for every sound file in Happy
        self.descriptorsNeutral = [] #audio descriptorss for every sound file in Neutral
        

    def _saveDescriptorsIntoMatrices(self):

        print "EmoMus5_Test_2s -> Saving descriptors into matrices" 

        for rownum in range(1,self.sh.nrows):
            row = self.sh.row_values(rownum)
            #print "Row Number: " + str(rownum)
            descriptorsArray = []
            
            for colnum in range(1,self.sh.ncols-1): #col=0 is the file name col=lastCol is the class
                #print row[colnum]
                descriptorsArray.append(float(row[colnum]))
            
            descriptorsArray = np.array(descriptorsArray)
            #descriptorsArray = descriptorsArray[indexes]
            descriptorsArray = np.nan_to_num(descriptorsArray)
            
            audio_class = str(row[self.sh.ncols-1])
            
            if(audio_class =='Anxious'): # the wav file corresponds to Anxious
                 #print "audio_class == Anxious"
                self.descriptorsAnxious.append(descriptorsArray)      
            elif(audio_class =='Happy'): # the wav file corresponds to Happy
                self.descriptorsHappy.append(descriptorsArray)
                #print "audio_class == Happy"
            elif(audio_class =='Neutral'): # the wav file corresponds to Neutral
                self.descriptorsNeutral.append(descriptorsArray)
                #print "audio_class == Neutral"
              
        self.descriptorsAnxious = np.array(self.descriptorsAnxious)
        self.descriptorsAnxious = np.vstack((self.descriptorsAnxious,self.descriptorsAnxious)) #the test was twice repeated with the same audio files
        self.descriptorsHappy = np.array(self.descriptorsHappy)
        self.descriptorsHappy = np.vstack((self.descriptorsHappy,self.descriptorsHappy)) #the test was twice repeated with the same audio files
        self.descriptorsNeutral = np.array(self.descriptorsNeutral)
        self.descriptorsNeutral = np.vstack((self.descriptorsNeutral,self.descriptorsNeutral)) #the test was twice repeated with the same audio files

        print "EmoMus5_Test_2s -> Anxious music descriptors dimensions: " + str(self.descriptorsAnxious.shape)
        print "EmoMus5_Test_2s -> Happy music descriptors dimensions: " + str(self.descriptorsHappy.shape)
        print "EmoMus5_Test_2s -> Neutral music descriptors dimensionspe: " + str(self.descriptorsNeutral.shape)

        self.numAnxtests = self.descriptorsAnxious.shape[0]
        self.numHaptests = self.descriptorsHappy.shape[0]
        self.numNeutests = self.descriptorsNeutral.shape[0]

        print "EmoMus5_Test_2s -> Anxious music descriptors number: " + str(self.numAnxtests)
        print "EmoMus5_Test_2s -> Happy music descriptors number: " + str(self.numHaptests)
        print "EmoMus5_Test_2s -> Neutral music descriptors number: " + str(self.numNeutests)

    def _createLeave2OutCrossValidationIndexes(self):

        print "EmoMus5_Test_2s -> Creating leave two out cross validation indexes " 

        for i in range(self.descriptorsAnxious.shape[1]):
            self.descriptorsAnxiousInd = []
            for n in range(self.numAnxtests):
                 ind = []
                 for m in range(self.numAnxtests):
                     if(not(m == n)):
                         ind.append(m)
                 self.descriptorsAnxiousInd.append(ind)

        self.descriptorsHappyInd = []
        for n in range(self.numHaptests):
             ind = []
             for m in range(self.numHaptests):
                 if(not(m == n)):
                     ind.append(m)
             self.descriptorsHappyInd.append(ind)

        self.descriptorsNeutralInd = []
        for n in range(self.numNeutests):
             ind = []
             for m in range(self.numNeutests):
                 if(not(m == n)):
                     ind.append(m)
             self.descriptorsNeutralInd.append(ind)      
      
           
    def _CalcPredictionModel(self, PM_method):

        if (PM_method == 'SVR_linear'):
            #svr = svm.SVR(kernel='linear', C=1e4)
            svr = svm.SVR(kernel='linear', gamma=0.1, C=1e4)
        elif (PM_method == 'SVR_poly'):
            #svr = svm.SVR(kernel='poly', C=1e4, degree=2,gamma=0.1) #polynomial kernel, degree 2.
            svr = svm.SVR(kernel='poly', degree=1) #polynomial kernel, degree 2.
        elif (PM_method == 'SVR_rbf'):
            svr = svm.SVR(kernel='rbf', C=1e4, gamma=0.1) # Radial Basis Function Kernel
       
        Anx_Accuracy = np.zeros((self.numAnxtests)) 
        Hap_Accuracy = np.zeros((self.numHaptests)) 
        Neu_Accuracy = np.zeros((self.numNeutests)) 

        self.Anxfeatures = np.zeros([self.numSelFeatures,self.numAnxtests])
        self.Hapfeatures = np.zeros([self.numSelFeatures,self.numHaptests])
        self.Neufeatures = np.zeros([self.numSelFeatures,self.numNeutests])

        total = self.numAnxtests*(self.numHaptests + self.numNeutests) + (self.numHaptests*self.numNeutests)

        for nAnx in range(self.numAnxtests): 
            percentage = 100.0*nAnx*(self.numHaptests + self.numNeutests)/total
            print "User " + str(self.user) + ": " + str(percentage) + "%"
            indAnx = self.descriptorsAnxiousInd[nAnx]#we use the rest of experiments as training data for the mth test
            for nHap in range(self.numHaptests): 
                percentage = 100.0*(nAnx+1)*(nHap)/total
                print "User " + str(self.user) + ": " +  str(percentage)  + "%"
                indHap = self.descriptorsHappyInd[nHap]#we use the rest of experiments as training data for the lth test
                wavFeatures = np.vstack((self.descriptorsAnxious[indAnx,:],self.descriptorsHappy[indHap,:],
                                         self.descriptorsNeutral))

                self.predictedAnxVolumes = np.zeros(self.numSelFeatures)
                self.predictedHapVolumes = np.zeros(self.numSelFeatures)
                for n in range(self.numSelFeatures):
                  k = self.f_ind[n]%self.lenY #y axis
                  rest = self.f_ind[n]/self.lenY 
                  j = rest%self.lenX #x axis
                  i = rest/self.lenX #z axis  
                  #print self.anxiousVolumes.shape
                  self.Anxfeatures[n,:] = self.anxiousVolumes[:,j,k,i] # f_ind
                  self.Hapfeatures[n,:] = self.happyVolumes[:,j,k,i] # f_ind 
                  self.Neufeatures[n,:] = self.neutralVolumes[:,j,k,i] # f_ind

                  meanFeatures = np.hstack((self.Anxfeatures[n,indAnx],self.Hapfeatures[n,indHap],
                              self.Neufeatures[n,:]))
                  
                  if(PM_method == 'LinearReg'):
                    regr = linear_model.LinearRegression() # self.wavFeatures is an n-by-p matrix of p predictors at each of n observations
                    regr.fit( wavFeatures,meanFeatures)
                    # multiple linear regression models by allowing the response variable to be a function of k explanatory variables
                    #print regr.coef_ #coefficients #p0 + p1 + ...pn
                    self.predictedAnxVolumes = regr.predict(self.descriptorsAnxious[nAnx,:]) # b0 + b1*p1 + ..bn*pn 
                    self.predictedHapVolumes = regr.predict(self.descriptorsHappy[nHap,:]) # b0 + b1*p1 + ..bn*pn
                  else:
                    svr.fit(wavFeatures, meanFeatures)
                    self.predictedAnxVolumes = svr.predict(self.descriptorsAnxious[nAnx,:]) 
                    self.predictedHapVolumes = svr.predict(self.descriptorsHappy[nHap,:])

                    #Calculate the accuracy
                    AnxrAnxp_dist = spatial.distance.cosine(self.Anxfeatures[:,nAnx],self.predictedAnxVolumes)#dist Anx real vs Anx Predicted
                    AnxrHapp_dist = spatial.distance.cosine(self.Anxfeatures[:,nAnx],self.predictedHapVolumes)#dist Anx real vs Hap Predicted  
                    HaprAnxp_dist = spatial.distance.cosine(self.Hapfeatures[:,nHap],self.predictedAnxVolumes)#dist Hap real vs Anx Predicted         
                    HaprHapp_dist = spatial.distance.cosine(self.Hapfeatures[:,nHap],self.predictedHapVolumes)#dist Hap real vs Hap Predicted
                    if((AnxrAnxp_dist+ HaprHapp_dist) < (AnxrHapp_dist + HaprAnxp_dist)):
                        Anx_Accuracy[nAnx]+=1
                        Hap_Accuracy[nHap]+=1


            for nNeu in range(self.numNeutests):
                percentage = 100.0*(nAnx+1)*(self.numHaptests + nNeu)/total
                print "User " + str(self.user) + ": " +  str(percentage)  + "%"

                indNeu = self.descriptorsNeutralInd[nNeu]
               
                wavFeatures = np.vstack((self.descriptorsAnxious[indAnx,:],self.descriptorsHappy,
                                       self.descriptorsNeutral[indNeu,:]))


                for n in range(self.numSelFeatures):
                  k = self.f_ind[n]%self.lenY #y axis
                  rest = self.f_ind[n]/self.lenY 
                  j = rest%self.lenX #x axis
                  i = rest/self.lenX #z axis  
                  #print self.anxiousVolumes.shape
                  self.Anxfeatures[n,:] = self.anxiousVolumes[:,j,k,i] # f_ind
                  self.Hapfeatures[n,:] = self.happyVolumes[:,j,k,i] # f_ind 
                  self.Neufeatures[n,:] = self.neutralVolumes[:,j,k,i] # f_ind

                  meanFeatures = np.hstack((self.Anxfeatures[n,indAnx],self.Hapfeatures[n,:],
                            self.Neufeatures[n,indNeu]))

                  if(PM_method == 'LinearReg'):
                    regr = linear_model.LinearRegression() # self.wavFeatures is an n-by-p matrix of p predictors at each of n observations
                    regr.fit( wavFeatures,meanFeatures)
                    # multiple linear regression models by allowing the response variable to be a function of k explanatory variables
                    #print regr.coef_ #coefficients #p0 + p1 + ...pn
                    self.predictedAnxVolumes = regr.predict(self.descriptorsAnxious[nAnx,:]) # b0 + b1*p1 + ..bn*pn 
                    self.predictedNeuVolumes = regr.predict(self.descriptorsNeutral[nNeu,:]) # b0 + b1*p1 + ..bn*pn
                  else:
                    svr.fit(wavFeatures, meanFeatures)
                    self.predictedAnxVolumes = svr.predict(self.descriptorsAnxious[nAnx,:]) 
                    self.predictedNeuVolumes = svr.predict(self.descriptorsNeutral[nNeu,:])

                    #Calculate the accuracy
                    AnxrAnxp_dist = spatial.distance.cosine(self.Anxfeatures[:,nAnx],self.predictedAnxVolumes)#dist Anx real vs Anx Predicted
                    AnxrNeup_dist = spatial.distance.cosine(self.Anxfeatures[:,nAnx],self.predictedNeuVolumes)#dist Anx real vs Neu Predicted  
                    NeurAnxp_dist = spatial.distance.cosine(self.Neufeatures[:,nNeu],self.predictedAnxVolumes)#dist Neu real vs Anx Predicted         
                    NeurNeup_dist = spatial.distance.cosine(self.Neufeatures[:,nNeu],self.predictedNeuVolumes)#dist Neu real vs Neu Predicted
                    if((AnxrAnxp_dist+ NeurNeup_dist) < (AnxrNeup_dist + NeurAnxp_dist)):
                        Anx_Accuracy[nAnx]+=1
                        Neu_Accuracy[nNeu]+=1

        for nHap in range(self.numHaptests): 
            percentage = 100.0*(self.numAnxtests*(self.numHaptests + self.numNeutests) + nHap*self.numNeutests)/total
            print "User " + str(self.user) + ": " + str(percentage) + "%"
            indHap = self.descriptorsHappyInd[nHap]#we use the rest of experiments as training data for the mth test
            
            for nNeu in range(self.numNeutests): 
                percentage = 100.0*(self.numAnxtests*(self.numHaptests + self.numNeutests)/total + (nHap+1)*nNeu)/total
                print "User " + str(self.user) + ": " + str(percentage) + "%"
                indNeu = self.descriptorsNeutralInd[nNeu]#we use the rest of experiments as training data for the lth test
                
                wavFeatures = np.vstack((self.descriptorsAnxious,self.descriptorsHappy[indHap,:],
                                       self.descriptorsNeutral[indNeu,:]))


                for n in range(sself.numSelFeatures):
                    k = self.f_ind[n]%self.lenY #y axis
                    rest = self.f_ind[n]/self.lenY 
                    j = rest%self.lenX #x axis
                    i = rest/self.lenX #z axis  
                    #print self.anxiousVolumes.shape
                    self.Anxfeatures[n,:] = self.anxiousVolumes[:,j,k,i] # f_ind
                    self.Hapfeatures[n,:] = self.happyVolumes[:,j,k,i] # f_ind 
                    self.Neufeatures[n,:] = self.neutralVolumes[:,j,k,i] # f_ind

                    meanFeatures = np.hstack((self.Anxfeatures[n,:],self.Hapfeatures[n,indHap],
                            self.Neufeatures[n,indNeu]))

                    if(PM_method == 'LinearReg'):
                      regr = linear_model.LinearRegression() # self.wavFeatures is an n-by-p matrix of p predictors at each of n observations
                      regr.fit( wavFeatures,meanFeatures)
                      # multiple linear regression models by allowing the response variable to be a function of k explanatory variables
                      #print regr.coef_ #coefficients #p0 + p1 + ...pn
                      self.predictedHapVolumes = regr.predict(self.descriptorsHappy[nHap,:]) # b0 + b1*p1 + ..bn*pn 
                      self.predictedNeuVolumes = regr.predict(self.descriptorsNeutral[nNeu,:]) # b0 + b1*p1 + ..bn*pn
                    else:
                      svr.fit(wavFeatures, meanFeatures)
                      self.predictedHapVolumes = svr.predict(self.descriptorsHappy[nHap,:]) 
                      self.predictedNeuVolumes = svr.predict(self.descriptorsNeutral[nNeu,:])

                      #Calculate the accuracy
                      HaprHapp_dist = spatial.distance.cosine(self.Hapfeatures[:,nHap],self.predictedHapVolumes)#dist Hap real vs Hap Predicted
                      HaprNeup_dist = spatial.distance.cosine(self.Hapfeatures[:,nHap],self.predictedNeuVolumes)#dist Hap real vs Neu Predicted  
                      NeurHapp_dist = spatial.distance.cosine(self.Neufeatures[:,nNeu],self.predictedHapVolumes)#dist Neu real vs Hap Predicted         
                      NeurNeup_dist = spatial.distance.cosine(self.Neufeatures[:,nNeu],self.predictedNeuVolumes)#dist Neu real vs Neu Predicted
                      if((HaprHapp_dist+ NeurNeup_dist) < (HaprNeup_dist + NeurHapp_dist)):
                          Neu_Accuracy[nNeu]+=1
                          Hap_Accuracy[nHap]+=1
              
        
        Anx_Accuracy/=(self.numHaptests + self.numNeutests )
        Hap_Accuracy/=(self.numAnxtests + self.numNeutests )
        Neu_Accuracy/=(self.numHaptests + self.numAnxtests )
                
        self.PMaccuracy = [] 
        self.PMaccuracy.append(np.mean(Anx_Accuracy)*100)
        self.PMaccuracy.append(np.mean(Hap_Accuracy)*100)
        self.PMaccuracy.append(np.mean(Neu_Accuracy)*100)
   
    






    def _readFeatures(self,filename, separator=','):
        """ Read a file with an arbitrary number of columns.
            The type of data in each column is arbitrary
            It will be cast to the given dtype at runtime
        """
        cast = np.cast
        data = []
        for line in open(filename, 'r'):
            fields = line.strip().split(separator)
            dataRow = []
            for i, number in enumerate(fields):
                dataRow.append(float(number))
            data.append(dataRow)
        return np.array(data) 
      
    def _SVM_Analysis(self):
        
        print "SVM Analysis"
        
        TasksInd = np.array(self.emotions[0])
        Y_Anova = np.zeros((len(self.emotions[0]))) #array-like, shape = [n_samples]
        for i in range(1,len(self.emotions)):
            Y_Anova = np.hstack((Y_Anova,i*np.ones((len(self.emotions[i])))))
            TasksInd = np.hstack((TasksInd,np.array(self.emotions[i])))
        
        t_ind12 = self.t_ind[0] #t-test for tasks 1&2. Duration Beat vs Duration NonBeat
        X_ttest = np.zeros((TtestTasksInd.size,t_ind12.size))
        for n,feature in enumerate(t_ind12):
            if(n%50000==0):
                print "SVM Feature number: ", n
            # To reconstruct from indexes
            k = feature%self.lenY #y axis
            rest = feature/self.lenY 
            j = rest%self.lenX #x axis
            i = rest/self.lenX #z axis 
            X_ttest[:,n] =  self.data[TtestTasksInd,i,j,k]
        
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_ttest, Y_ttest) # Fit the linear SVM model according to the given training data and parameters.
                                    # X = Training vectors [n_samples, n_features], Y = Target values [n_samples]
        W_ttest = lin_clf.coef_ # Weights assigned to the features 
                                #  array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
        W_ttest = W_ttest.flatten()
        W_ttest = abs(W_ttest) #Should we use the absolute value??
        W_ttest_ind = W_ttest.ravel().argsort() # Sort, return indices
        W_ttest_ind = W_ttest_ind[::-1] # reverse (from greater values to lower values)
        self.W_ttest = t_ind12[W_ttest_ind] #we store the new indices from original ones
        
    def _saveSelectedFeatures(self): 
        fi = open(self.path + "/Info/selectedFeatures_" + self.user + ".dat", 'w')
        pickle.dump(self.f_ind, fi,protocol= pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.emotions, fi,protocol= pickle.HIGHEST_PROTOCOL)
        fi.close()
        # To reconstruct, then 
        # k = t_array_ind(i)%y, rest = t_array_ind(i)/y  j = rest%x , i = rest/x
        
    def _writePM_Results(self,PM_method,dist): 
        indPM = self.PM_list.index(PM_method)
        indFL = self.numFeat_list.index(self.numSelFeatures)
        indDist = self.distance_list.index(dist) 
        self.ws.write(2+(len(self.test_List)+3)*indDist, indFL+1+len(self.numFeat_list)*indPM,  'n = ' + str(self.numSelFeatures)) 
        for n, test in enumerate(self.test_List):
            strAcc = "%.2f" % self.PMaccuracy[n] + "%"
            self.ws.write(n+3+(len(self.test_List)+3)*indDist, indFL+1+len(self.numFeat_list)*indPM, strAcc) 

    def _dimReduction(self):
        
        #ANOVA Analysis
        f_list = []
        np.seterr(invalid='ignore')# what do we do if var = 0??
        meanData = self.data.mean(axis=0)
        meanData[meanData<350]=0; #Background Filtering
        f_array =  np.zeros((self.lenX*self.lenY*self.lenZ))
        n=-1
        for i in range(self.lenZ):   
            print "DimReduction ", i       
            for j in range(self.lenX):
                for k in range(self.lenY):
                    n+=1
                    if(meanData[i,j,k]>0):
                        [f_value, prob] = stats.f_oneway(self.data[j,k,i,self.emotions[0]],self.data[j,k,i,self.emotions[1]],self.data[j,k,i,self.emotions[2]])
                        if (not np.isnan(f_value)):
                            f_array[n]= np.abs(f_value) #ANOVA
        

        self.f_ind = f_array.ravel().argsort() # Sort, return indices
        self.f_ind = self.f_ind[::-1] # reverse
        self.f_ind = self.f_ind[f_array[self.f_ind].ravel().nonzero()] #we erase all zero values
            
                    
    def _saveData(self):
       #np.save('fMRI_Data.npy', self.data)
        print "saving Data to file"
        fi = open(self.path + "fMRI_Data.dat","w")
        pickle.dump(self.data, fi,protocol= pickle.HIGHEST_PROTOCOL)
        fi.close()
        print "Data saved"
    
    def _gaussKern(self, size):
        """ Returns a normalized 2D gaussian kernel array for convolutions """
        size = int(size)
        x, y, z = np.mgrid[-size:size+1, -size:size+1, -size:size+1]
        g = np.exp(-(x**2/float(size) + y**2/float(size) + z**2/float(size)))
        return g / g.sum()

    def _smoothData(self,data, n) :
        #print data.max()
        """ smooths the data by convolving with a gaussian kernel of typical size n."""
        (z,x,y) = np.shape(data) 
        #x_gauss =  np.zeros((z+2*n,x+2*n,y+2*n))
        #x_gauss[n:z+n-1,n:x+n-1,n:y+n-1] = data.copy()
        g = self._gaussKern(n)
        data_proc = signal.convolve(data, g, mode='same')
        #print data_proc.max()
        return(data_proc)
  